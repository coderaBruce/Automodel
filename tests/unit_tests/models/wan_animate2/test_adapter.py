# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for the Wan-Animate-2 flow-matching adapter.

Every test here is CPU-only and stubs the transformer: the upstream
``WanAnimate2Transformer3DModel`` attention kernels hard-assert
``q.device.type == "cuda"``, so no real forward pass can run on CPU.
``WanAnimate2Adapter.forward`` also resolves the attention backend through
``maybe_patch_flash_attention``, which raises unless the Wan-Animate-2 diffusers
build is importable; the :func:`attention_patch_calls` fixture stubs that hook so
nothing here depends on the fork being installed.

The geometry expectations come from how the upstream transformer *consumes* the
adapter's outputs: its ``(1, 2, 2)`` patch embedding turns every latent frame
into ``(latent_height // 2) * (latent_width // 2)`` tokens, and
``WanAnimate2Transformer3DModel.create_mask`` recovers the driving latent frame
count as ``origin_len // 4 + 1`` and the per-frame token count as
``prod(origin_area) // 256``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.flow_matching.pipeline import create_adapter
from nemo_automodel.components.models.wan_animate2 import adapter as adapter_module
from nemo_automodel.components.models.wan_animate2.adapter import WanAnimate2Adapter

TARGET_FRAMES = 3
LATENT_HEIGHT = 4
LATENT_WIDTH = 6
LATENT_CHANNELS = 16
CONDITIONING_CHANNELS = 20
MASK_CHANNELS = 4
TEXT_TOKENS = 5
REF_TEXT_TOKENS = 4
CLIP_TOKENS = 257
CLIP_DIM = 1280
TEXT_DIM = 4096
# The upstream (1, 2, 2) patch embedding halves both spatial axes.
TOKENS_PER_FRAME = (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)


@pytest.fixture(autouse=True)
def attention_patch_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the attention-backend hook that :meth:`WanAnimate2Adapter.forward` resolves once.

    ``forward`` calls ``maybe_patch_flash_attention``, which raises ``ImportError``
    unless ``diffusers.models.transformers.transformer_wan_animate_2`` is
    importable. That Wan-Animate-2 diffusers build is not a test dependency, so
    the hook is replaced by a recorder. The process-global "already resolved"
    flag is reset alongside it, which keeps the tests order-independent.

    Returns:
        List receiving one entry per stubbed invocation.
    """
    calls: list[str] = []

    def _record() -> bool:
        calls.append("maybe_patch_flash_attention")
        return False

    monkeypatch.setattr(adapter_module, "maybe_patch_flash_attention", _record)
    monkeypatch.setattr(adapter_module, "_ATTENTION_PATCH_CHECKED", False)
    return calls


def _make_batch(
    *,
    batch_size: int = 1,
    driving_frames: int = TARGET_FRAMES,
    latent_height: int = LATENT_HEIGHT,
    latent_width: int = LATENT_WIDTH,
) -> dict[str, torch.Tensor]:
    """Build a cached Wan-Animate-2 triplet batch of tiny tensors.

    Args:
        batch_size: Size of axis 0 for every tensor in the batch.
        driving_frames: Latent frame count of the driving video.
        latent_height: Latent grid height.
        latent_width: Latent grid width.

    Returns:
        Mapping of the cache keys the adapter reads: the four latent entries are
        [batch, 16, frames, latent_height, latent_width] with TARGET_FRAMES
        frames (one for ``reference_latents``, ``driving_frames`` for
        ``driving_latents``), the two CLIP entries [batch, 257, 1280] and the two
        text entries [batch, tokens, 4096].
    """
    return {
        "video_latents": torch.randn(batch_size, LATENT_CHANNELS, TARGET_FRAMES, latent_height, latent_width),
        "reference_latents": torch.randn(batch_size, LATENT_CHANNELS, 1, latent_height, latent_width),
        "driving_latents": torch.randn(batch_size, LATENT_CHANNELS, driving_frames, latent_height, latent_width),
        "cond_zero_latents": torch.zeros(batch_size, LATENT_CHANNELS, TARGET_FRAMES, latent_height, latent_width),
        "clip_fea": torch.randn(batch_size, CLIP_TOKENS, CLIP_DIM),
        "clip_fea_ref": torch.randn(batch_size, CLIP_TOKENS, CLIP_DIM),
        "text_embeddings": torch.randn(batch_size, TEXT_TOKENS, TEXT_DIM),
        "prompt_ref_embeddings": torch.randn(batch_size, REF_TEXT_TOKENS, TEXT_DIM),
    }


def _make_context(
    batch: dict[str, torch.Tensor],
    *,
    sigma_value: float = 0.5,
    dtype: torch.dtype = torch.float32,
) -> FlowMatchingContext:
    """Wrap a cached batch in a CPU flow-matching context.

    Args:
        batch: Cached batch laid out as documented by :func:`_make_batch`.
        sigma_value: Flow-matching noise level used for every sample.
        dtype: Compute dtype the adapter must cast its outputs to. The cached
            tensors stay float32, matching a cache read back for bf16 training.

    Returns:
        Context whose float32 ``noisy_latents`` and ``latents`` have the shape of
        ``video_latents`` and whose ``timesteps`` and ``sigma`` have shape
        [batch].
    """
    latents = batch["video_latents"]
    batch_size = latents.shape[0]
    sigma = torch.full((batch_size,), sigma_value)
    noise = torch.randn_like(latents)
    broadcast_sigma = sigma.view(batch_size, 1, 1, 1, 1)
    return FlowMatchingContext(
        noisy_latents=(1.0 - broadcast_sigma) * latents + broadcast_sigma * noise,
        latents=latents,
        timesteps=sigma * 1000.0,
        sigma=sigma,
        task_type="i2v",
        data_type="video",
        device=torch.device("cpu"),
        dtype=dtype,
        batch=batch,
    )


@dataclass
class _RecordedCall:
    """One recorded call into :class:`_RecordingTransformer`, as observed on entry."""

    method: str
    grad_enabled: bool
    # First positional argument: per-sample tensors of shape
    # [channels, frames, latent_height, latent_width].
    stream: list[torch.Tensor]
    # Every other upstream keyword, including the ``_ref`` / non-``_ref`` pairs.
    kwargs: dict[str, Any]
    key_cache: dict[int, torch.Tensor]
    value_cache: dict[int, torch.Tensor]
    cached_entries_on_entry: int


class _RecordingTransformer(nn.Module):
    """Stub for the ``method``-dispatched upstream transformer.

    The reference phase writes one key/value tensor per block into the caches and
    returns nothing; the generation phase scales its input stream by a trainable
    parameter and returns the per-sample list the adapter expects.
    """

    def __init__(self, *, num_blocks: int = 2, scale: float = 2.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.num_blocks = num_blocks
        self.calls: list[_RecordedCall] = []

    def forward(self, stream: list[torch.Tensor], **kwargs: Any) -> list[torch.Tensor] | None:
        """Record the call and emulate the two upstream phases.

        Args:
            stream: Per-sample tensors of shape [channels, frames,
                latent_height, latent_width].
            **kwargs: Upstream keywords, including ``method``, ``k_cache`` and
                ``v_cache``.

        Returns:
            ``None`` for ``forward_ref``; for ``forward_gen`` a list of tensors
            of shape [16, frames, latent_height, latent_width], one per sample.
        """
        key_cache, value_cache = kwargs["k_cache"], kwargs["v_cache"]
        self.calls.append(
            _RecordedCall(
                method=kwargs["method"],
                grad_enabled=torch.is_grad_enabled(),
                stream=stream,
                kwargs=kwargs,
                key_cache=key_cache,
                value_cache=value_cache,
                cached_entries_on_entry=len(key_cache),
            )
        )
        if kwargs["method"] == "forward_ref":
            for index in range(self.num_blocks):
                key_cache[index] = stream[0].flatten() * self.scale
                value_cache[index] = stream[0].flatten() + self.scale
            return None
        return [sample * self.scale for sample in stream]


def test_prepare_inputs_stream_shapes_and_channel_layout() -> None:
    """The generation stream gains one frame; conditioning is 4 mask + 16 latent."""
    torch.manual_seed(0)
    batch = _make_batch()
    context = _make_context(batch)

    inputs = WanAnimate2Adapter().prepare_inputs(context)

    generation_stream = inputs["x"][0]
    conditioning = inputs["y"][0]
    driving_stream = inputs["x_ref"][0]
    driving_conditioning = inputs["condition_y"][0]

    assert len(inputs["x"]) == 1
    assert generation_stream.shape == (LATENT_CHANNELS, TARGET_FRAMES + 1, LATENT_HEIGHT, LATENT_WIDTH)
    assert conditioning.shape == (CONDITIONING_CHANNELS, TARGET_FRAMES + 1, LATENT_HEIGHT, LATENT_WIDTH)
    # The driving stream carries no reference slot, so it keeps its own frames.
    assert driving_stream.shape == (LATENT_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    assert driving_conditioning.shape == (CONDITIONING_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)

    # Only the leading (reference-slot) frame is new; the rest is the noisy target.
    torch.testing.assert_close(generation_stream[:, 1:], context.noisy_latents[0])
    torch.testing.assert_close(
        conditioning[MASK_CHANNELS:],
        torch.cat([batch["reference_latents"][0], batch["cond_zero_latents"][0]], dim=1),
    )
    torch.testing.assert_close(driving_conditioning[MASK_CHANNELS:], batch["driving_latents"][0])

    # Text and CLIP conditioning must not be swapped between the two phases.
    torch.testing.assert_close(inputs["context"][0], batch["text_embeddings"][0])
    torch.testing.assert_close(inputs["context_ref"][0], batch["prompt_ref_embeddings"][0])
    torch.testing.assert_close(inputs["clip_fea"], batch["clip_fea"])
    torch.testing.assert_close(inputs["clip_fea_ref"], batch["clip_fea_ref"])
    torch.testing.assert_close(inputs["timestep"], context.timesteps)


def test_prepare_inputs_conditioning_mask_marks_only_the_reference_frame() -> None:
    """The generation mask is on for the reference slot and off for target frames."""
    torch.manual_seed(1)
    context = _make_context(_make_batch())

    inputs = WanAnimate2Adapter().prepare_inputs(context)

    mask = inputs["y"][0][:MASK_CHANNELS]
    assert torch.equal(mask[:, 0], torch.ones(MASK_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH))
    assert torch.equal(mask[:, 1:], torch.zeros(MASK_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH))
    # Every driving frame is clean conditioning, so its mask is fully on.
    driving_mask = inputs["condition_y"][0][:MASK_CHANNELS]
    assert torch.equal(driving_mask, torch.ones(MASK_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH))


def test_prepare_inputs_geometry_matches_the_upstream_patch_embedding_and_block_mask() -> None:
    """Sequence lengths, the reference grid and the origin metadata agree with upstream."""
    torch.manual_seed(2)
    context = _make_context(_make_batch())

    inputs = WanAnimate2Adapter().prepare_inputs(context)

    # The patch embedding consumes 16 latent + 20 conditioning channels and emits
    # one token per (1, 2, 2) patch of the concatenated stream.
    assert inputs["x"][0].shape[0] + inputs["y"][0].shape[0] == 36
    assert inputs["seq_len"] == (TARGET_FRAMES + 1) * TOKENS_PER_FRAME
    assert inputs["seq_len_ref"] == TARGET_FRAMES * TOKENS_PER_FRAME
    # grid_sizes_ref indexes the reference RoPE tables, so it is the post-patch grid.
    assert inputs["grid_sizes_ref"].dtype == torch.long
    assert inputs["grid_sizes_ref"].tolist() == [[TARGET_FRAMES, LATENT_HEIGHT // 2, LATENT_WIDTH // 2]]

    # WanAnimate2Transformer3DModel.create_mask recovers the driving frame count
    # and per-frame token count from origin_len/origin_area, then sizes the query
    # stream as (origin_latent_f + 1) * hw against a key stream of
    # origin_latent_f * hw. origin_area is the pixel resolution behind the latent
    # grid (8x VAE).
    origin_latent_frames = inputs["origin_len"] // 4 + 1
    tokens_per_frame = math.prod(inputs["origin_area"]) // 256
    assert origin_latent_frames == TARGET_FRAMES
    assert tokens_per_frame == TOKENS_PER_FRAME
    assert (origin_latent_frames + 1) * tokens_per_frame == inputs["seq_len"]
    assert origin_latent_frames * tokens_per_frame == inputs["seq_len_ref"]
    assert inputs["origin_area"] == [LATENT_HEIGHT * 8, LATENT_WIDTH * 8]


def test_prepare_inputs_reference_slot_lies_on_the_flow_matching_path() -> None:
    """The reference slot interpolates the clean reference latent with noise."""
    adapter = WanAnimate2Adapter()
    torch.manual_seed(4)
    batch = _make_batch()
    reference_latent = batch["reference_latents"][0][:, 0]

    torch.manual_seed(5)
    clean_slot = adapter.prepare_inputs(_make_context(batch, sigma_value=0.0))["x"][0][:, 0]
    torch.manual_seed(5)
    noise_slot = adapter.prepare_inputs(_make_context(batch, sigma_value=1.0))["x"][0][:, 0]
    torch.manual_seed(5)
    half_slot = adapter.prepare_inputs(_make_context(batch, sigma_value=0.5))["x"][0][:, 0]

    # sigma=0 is the clean reference frame, sigma=1 carries none of it, and the
    # midpoint is the average of the two endpoints.
    torch.testing.assert_close(clean_slot, reference_latent)
    assert not torch.allclose(noise_slot, reference_latent)
    torch.testing.assert_close(half_slot, 0.5 * clean_slot + 0.5 * noise_slot)


def test_prepare_inputs_casts_every_model_input_to_the_context_dtype() -> None:
    """The cache is float32 while training runs in bf16, so every input is cast."""
    torch.manual_seed(21)
    context = _make_context(_make_batch(), dtype=torch.bfloat16)

    inputs = WanAnimate2Adapter().prepare_inputs(context)

    for key in ("x", "y", "x_ref", "condition_y", "context", "context_ref"):
        assert {tensor.dtype for tensor in inputs[key]} == {torch.bfloat16}, key
    for key in ("clip_fea", "clip_fea_ref", "timestep"):
        assert inputs[key].dtype == torch.bfloat16, key
    # grid_sizes_ref indexes the reference RoPE tables and must stay integral.
    assert inputs["grid_sizes_ref"].dtype == torch.long


def test_prepare_inputs_rejects_batch_size_greater_than_one() -> None:
    """Batched key/value packing is unsupported upstream, so batch > 1 must fail."""
    torch.manual_seed(6)
    context = _make_context(_make_batch(batch_size=2))

    with pytest.raises(ValueError, match="local_batch_size=1"):
        WanAnimate2Adapter().prepare_inputs(context)


def test_prepare_inputs_rejects_driving_latents_with_a_foreign_frame_count() -> None:
    """A driving clip of another length silently mis-sizes the upstream block mask.

    ``create_mask`` derives the flex-attention query span from the *driving*
    frame count while ``forward_gen`` embeds ``target_latent_frames + 1`` frames.
    A mismatch must be rejected here rather than produce a ``seq_len`` that
    disagrees with the block mask.
    """
    torch.manual_seed(7)
    batch = _make_batch(driving_frames=TARGET_FRAMES - 1)

    with pytest.raises(ValueError, match="same latent frame count"):
        WanAnimate2Adapter().prepare_inputs(_make_context(batch))


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (lambda batch: batch.pop("clip_fea_ref"), TypeError, "clip_fea_ref"),
        (
            lambda batch: batch.update(text_embeddings=batch["text_embeddings"].unsqueeze(-1)),
            ValueError,
            "'text_embeddings' must have 3 dimensions",
        ),
        (
            lambda batch: batch.update(driving_latents=torch.cat([batch["driving_latents"]] * 2)),
            ValueError,
            "'driving_latents' must have batch size 1",
        ),
    ],
    ids=["missing", "wrong-rank", "foreign-batch-size"],
)
def test_prepare_inputs_rejects_a_malformed_cached_tensor(
    mutate: Callable[[dict[str, torch.Tensor]], object], error: type[Exception], message: str
) -> None:
    """One shared check requires every cached key, its rank, and its batch size."""
    torch.manual_seed(8)
    batch = _make_batch()
    mutate(batch)

    with pytest.raises(error, match=message):
        WanAnimate2Adapter().prepare_inputs(_make_context(batch))


@pytest.mark.parametrize(
    ("key", "shape", "message"),
    [
        ("reference_latents", (1, LATENT_CHANNELS, 2, LATENT_HEIGHT, LATENT_WIDTH), "exactly one latent frame"),
        (
            "cond_zero_latents",
            (1, LATENT_CHANNELS, TARGET_FRAMES + 1, LATENT_HEIGHT, LATENT_WIDTH),
            "must match the target latent frame count",
        ),
        ("reference_latents", (1, LATENT_CHANNELS, 1, LATENT_HEIGHT + 2, LATENT_WIDTH), "spatial dims"),
        ("driving_latents", (1, LATENT_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH + 2), "spatial dims"),
    ],
)
def test_prepare_inputs_rejects_cached_geometry_that_disagrees_with_the_target(
    key: str, shape: tuple[int, ...], message: str
) -> None:
    """Cached conditioning is concatenated with the target, so its geometry must align."""
    torch.manual_seed(11)
    batch = _make_batch()
    batch[key] = torch.randn(shape)

    with pytest.raises(ValueError, match=message):
        WanAnimate2Adapter().prepare_inputs(_make_context(batch))


@pytest.mark.parametrize(
    ("mangle", "message"),
    [
        (lambda latents: latents[:, :, 0], "noisy latents"),
        (lambda latents: latents[:, :8], "16 latent channels"),
    ],
    ids=["image-shaped", "half-the-latent-channels"],
)
def test_prepare_inputs_rejects_noisy_latents_with_a_foreign_layout(
    mangle: Callable[[torch.Tensor], torch.Tensor], message: str
) -> None:
    """A video/image mix-up or a non-Wan VAE cache must fail loudly."""
    torch.manual_seed(12)
    context = _make_context(_make_batch())
    context.noisy_latents = mangle(context.noisy_latents)

    with pytest.raises(ValueError, match=message):
        WanAnimate2Adapter().prepare_inputs(context)


def test_prepare_inputs_rejects_a_latent_grid_the_patch_size_cannot_tile() -> None:
    """An odd latent height cannot be tiled by the (1, 2, 2) patch embedding."""
    torch.manual_seed(13)
    batch = _make_batch(latent_height=LATENT_HEIGHT + 1)

    with pytest.raises(ValueError, match=r"divisible by the \(2, 2\) patch size"):
        WanAnimate2Adapter().prepare_inputs(_make_context(batch))


def test_forward_runs_the_reference_pass_first_with_an_isolated_fresh_cache(
    attention_patch_calls: list[str],
) -> None:
    """Two calls per step, reference first, each fed its own conditioning from a fresh no-grad cache.

    Args:
        attention_patch_calls: Recorder from the stubbed attention-backend hook.
    """
    torch.manual_seed(16)
    adapter = WanAnimate2Adapter()
    model = _RecordingTransformer()
    inputs = adapter.prepare_inputs(_make_context(_make_batch()))

    adapter.forward(model, inputs)
    adapter.forward(model, inputs)

    # The attention backend is resolved once per process, not once per step.
    assert attention_patch_calls == ["maybe_patch_flash_attention"]
    assert [call.method for call in model.calls] == [
        "forward_ref",
        "forward_gen",
        "forward_ref",
        "forward_gen",
    ]
    first_reference, first_generation, second_reference, _ = model.calls

    # Each phase must be handed its own stream and its own conditioning. The two
    # passes take near-identical keywords that differ only by a ``_ref`` suffix,
    # so a swap would still run and would silently train the driving branch on
    # the target's conditioning.
    assert first_reference.stream is inputs["x_ref"]
    assert first_reference.kwargs["y_ref"] is inputs["condition_y"]
    assert first_reference.kwargs["context_ref"] is inputs["context_ref"]
    assert first_reference.kwargs["clip_fea_ref"] is inputs["clip_fea_ref"]
    assert first_reference.kwargs["seq_len_ref"] == inputs["seq_len_ref"]
    assert first_generation.stream is inputs["x"]
    assert first_generation.kwargs["y"] is inputs["y"]
    assert first_generation.kwargs["context"] is inputs["context"]
    assert first_generation.kwargs["clip_fea"] is inputs["clip_fea"]
    assert first_generation.kwargs["seq_len"] == inputs["seq_len"]

    # The reference pass must not build autograd state.
    assert first_reference.grad_enabled is False
    assert first_generation.grad_enabled is True
    assert first_reference.key_cache[0].requires_grad is False

    # Within a step the generation pass reads exactly what the reference wrote.
    assert first_generation.key_cache is first_reference.key_cache
    assert first_generation.value_cache is first_reference.value_cache
    assert first_reference.cached_entries_on_entry == 0
    assert first_generation.cached_entries_on_entry == model.num_blocks

    # Across steps no cache state leaks.
    assert second_reference.key_cache is not first_reference.key_cache
    assert second_reference.value_cache is not first_reference.value_cache
    assert second_reference.cached_entries_on_entry == 0


def test_forward_drops_the_reference_slot_and_keeps_the_target_frames() -> None:
    """The prediction covers the target frames only and stays differentiable."""
    torch.manual_seed(18)
    adapter = WanAnimate2Adapter()
    model = _RecordingTransformer(scale=2.0)
    context = _make_context(_make_batch())
    inputs = adapter.prepare_inputs(context)

    prediction = adapter.forward(model, inputs)

    assert prediction.shape == (1, LATENT_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    # The stub scales its input, so the surviving frames must be the noisy target
    # frames: had the trailing frame been dropped instead, this would fail.
    torch.testing.assert_close(prediction, context.noisy_latents * 2.0)

    prediction.sum().backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad).all()


def test_create_adapter_resolves_the_wan_animate2_adapter() -> None:
    """The recipe-facing factory name maps to this adapter."""
    assert isinstance(create_adapter("wan_animate2"), WanAnimate2Adapter)
