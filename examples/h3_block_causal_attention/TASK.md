I am training a MiniMax-H3-style video model (`video/attention.py`) and the attention is the
bottleneck: it materialises the full score plane, and at real video lengths it does not fit. I
want `block_causal_attention` as one fused flash-style kernel, forward and backward.

```python
def block_causal_attention(q, k, v, block_size):
    # q, k, v: (B, H, S, D) bf16 with H = 56, D = 128; S = t * block_size packed tokens, frame by frame
    scores = (q @ k.transpose(-2, -1)).float() * (D ** -0.5)
    scores = scores.masked_fill(~block_causal_mask(S, block_size), float("-inf"))   # frame(q) >= frame(k)
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return probs @ v
```

The mask is causal at *frame* granularity and dense inside a frame: `frame(i) = i // block_size`
with `block_size = h * w` tokens per latent frame, and a token sees every token of its own frame
and of all earlier frames. Nothing is stored for it; it is a function of `S` and `block_size`.
The softmax is in fp32 (q, k, v are bf16 under autocast). `S` is a multiple of `block_size` in
the model, but please do not assume it: a partial last frame does occur in some of our data. Two
degenerate settings should also just work: `block_size >= S` (one frame: dense attention) and
`block_size = 1` (plain causal).

Training config is `video/settings.py` (`TrainConfig`, `ModelConfig`): bf16 autocast, batch 1, a
`(6, 16, 32)` grid = 6 frames of 512 tokens = 3072 packed tokens, `hidden_size` 5376, 56 heads
of dim 128, 1 layer. Real runs have `30 x 52 = 1560` tokens per frame and 14 to 40 frames (`S` =
21840 to 62400), where the eager code cannot even allocate the scores, so the kernel has to be
correct there too (that is where int32 indexing breaks: 56 * 6200 * 6200 score elements already
exceed 2^31). The training run is `python run.py --steps 50`; it prints `median_step_ms` and
`final_loss`.

Please take it all the way: spec, kernel with the backward (`dq`, `dk`, `dv`), verification and
benchmark against speed of light and against `torch.compile` and
`F.scaled_dot_product_attention` with the boolean mask, and integrate it into
`video/attention.py` behind a flag so I can switch back to the eager code. The q/k preparation
in front of it (`qk_prep`: per-head RMSNorm and 3D MM-RoPE) stays as it is; do not fuse it into
this kernel.

For a bounded correctness smoke, run `python run.py --smoke --steps 3 --seed 0` from
`user_repo/`. This keeps target channel/head dimensions but reduces batch/token counts; use the
unmodified representative config for performance measurements.
