# Debug Tools

Lead with **`TL_LOWER_TRACE`** when asking “which pass broke this?”. It
captures IR before and after every lowering pass, including codegen, and
writes a terminal and/or HTML report. Set it **before** `import tilelang`.

```bash
TL_LOWER_TRACE=1 python3 my_script.py              # HTML (default)
TL_LOWER_TRACE=terminal python3 my_script.py       # terminal only
TL_LOWER_TRACE=both python3 my_script.py           # both
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `TL_LOWER_TRACE` | `0`/`off`, `1`/`on`→html, `terminal`, `html`, `both` | off |
| `TL_LOWER_TRACE_DIR` | Artifact root | `./tmp/lower_trace_dir` |

Typical output: `<TL_LOWER_TRACE_DIR>/<script>/report.html` plus per-pass
`.tir` dumps and `codegen.cpp`. The HTML report is crash-safe (flushed after
each pass). Edit-and-recompile of generated codegen is supported.

Also `tilelang.PassConfigKey.TL_PASS_PROFILE` /
`TILELANG_PASS_PROFILE` for per-pass timing, with
`TL_PASS_PROFILE_THRESHOLD_MS` to hide fast passes.

## Pass Visualizer

For **structural** passes (layout inference, warp specialization,
pipelining), use the Pass Visualizer. It renders an `SBlock` tree rather
than a line diff:

```bash
python -m tilelang.tools.pass_visualizer.viewer \
    path/to/kernel.py \
    --set M=1024 --set N=1024 --set K=1024 \
    --out kernel_passes.html
```

## Legacy `TILELANG_PASS_DIFF`

`TILELANG_PASS_DIFF` (`0` / `terminal` / `html` / `both`) is the older
TVMScript line-diff hook. Prefer `TL_LOWER_TRACE`. Leave
`TILELANG_PASS_DIFF` off unless you need the old text-diff report
(`TILELANG_PASS_DIFF_OUTPUT`).

## Choose A Path

- Compile failure: `TL_LOWER_TRACE`, then inspect
  `kernel.get_kernel_source()`. Minimize with AutoDD if the source is large.
- Wrong result: compare a reference, add guarded `T.print(...)`, check
  indexing/copy boundaries.
- Layout / swizzle: `TL_LAYOUT_VISUALIZATION_ENABLE` or
  `tilelang.tools.plot_layout(...)`.
- Pass question: `TL_LOWER_TRACE` first; check `pass_configs` second.
- Performance: prove correctness, then profile. This page is not a
  profiler guide.

## Inspect Generated Source

```python
kernel = my_kernel.compile(...)
print(kernel.get_kernel_source())
```

Generated-source callbacks intercept source during compilation and affect
the whole process:

```python
tilelang.register_cuda_postproc(lambda code, target: (print(code), code)[1])
```

IR dumps:

```python
@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_DUMP_IR: True,
    tilelang.PassConfigKey.TL_DUMP_IR_DIR: "./dump_ir",
})
```

## Runtime Prints

Use `T.print(...)` only in small reproductions or narrow branches:

```python
if bx == 0 and by == 0:
    T.print("value", C_local[0, 0])
```

## Minimize With AutoDD

```bash
python -m tilelang.autodd tilelang_buggy.py \
  --err-msg "Dimension mismatch" \
  -o minimized.py \
  -j 4
```

`--err-msg` is a stable stdout/stderr substring. Freeze setup with
`from tilelang.autodd import __freeze__` or `# autodd: freeze-start` /
`# autodd: end-freeze`.

## Checklist

1. Reproduce with `TILELANG_DISABLE_CACHE=1` (and
   `TILELANG_AUTO_TUNING_DISABLE_CACHE=1` if tuning).
2. Validate against a small reference.
3. Set `TL_LOWER_TRACE=1` and inspect the HTML report / generated source.
4. Add guarded `T.print(...)` only when runtime values matter.
5. Use AutoDD once the failure has a stable command and error substring.
6. Record the minimized source and error in `FAQs.md` if it is likely to
   recur.
