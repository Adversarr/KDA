# Synthetic segmented multi-view attention

A small eager PyTorch training fixture with alternating frame/global attention, trainable
QKV/projection layers, QK normalization, and prefix lengths supplied as CPU metadata.
No KDA package, datasets or pretrained weights are needed. This is not the official VGGT
implementation. No positional encoder is included in this synthetic attention fixture.

Run from this directory on CUDA:

```bash
python -m segmented.train --smoke --steps 3 --seed 0
python -m segmented.train --steps 50 --seed 0
```

The full representative shape uses four views of 1374 tokens and 16 heads of width64.
`--smoke` reduces token counts only. Output includes per-step loss/time, followed by `median_step_ms` and `final_loss`.
Attention computes all queries; the loss excludes padded positions via prefix slices.

## Official implementation check


Reviewed `facebookresearch/vggt` main at commit
[`a288dd0f14786c93483e45524328726ab7b1b4ce`](https://github.com/facebookresearch/vggt/tree/a288dd0f14786c93483e45524328726ab7b1b4ce):

- [`layers/attention.py`](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/vggt/layers/attention.py): the default fused attention calls SDPA without a key-validity mask; the padding/length metadata here remains a synthetic extension.
- [`models/aggregator.py`](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/vggt/models/aggregator.py): alternates frame `(B*S,P,C)` and global `(B,S*P,C)` attention with camera/register tokens and positions.
- [`utils/load_fn.py`](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/vggt/utils/load_fn.py): resizes/crops/pads images but does not return the per-token attention-validity mask described by KDA's padded fixture.
- [`training/data/dynamic_dataloader.py`](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/training/data/dynamic_dataloader.py): synchronizes image count and aspect ratio within a batch.

KDA's existing `vggt_padded_attention` adds a synthetic padding mask and FlashAttention-style
mixed-precision semantics. This new example preserves **that extension's** contract while
changing how valid keys are represented. A spatial image rectangle does not automatically
become a token prefix; the producer must supply an actually prefix-ordered token layout.
No official VGGT code is copied into this fixture.

On 2026-09-08 the attention contract adopted FlashAttention-style operand rounding:
softmax stays FP32, its weights round to the input dtype before PV, and PV accumulates
in FP32. This supersedes the earlier FP32-probability requirement, while preserving
the synthetic padding semantics and numerical tolerances.

The mixed-precision eager path is validated with the repository CUDA/PyTorch 2.11
image. It uses `torch.bmm(..., out_dtype=torch.float32)` for low-precision
operands with FP32 product outputs, and an explicit Flash-style adjoint.
