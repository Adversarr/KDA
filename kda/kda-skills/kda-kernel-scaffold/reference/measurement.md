# Measurement scope

`measure()` returns raw samples, actual method, cache policy, synchronization and metric.
`bench_ms()` remains the legacy cold-cache median wrapper. `profiler` sums device activity,
`events` measures the current stream interval, and `wall` synchronizes the selected device
around the callable. None establishes CPU overhead by subtraction. Device sums can exceed
elapsed time when work overlaps. Multi-stream work requires explicit dependencies; request
drivers must wait for every rank before returning.

The operation runner supports cold or warm measurements. Context measurements require a
real producer/consumer driver and `context={producer, consumer, configuration}`. Roof
comparison is available only for cold profiler samples matching its calibration. An auto
fallback records its actual method; never compare mixed methods as one experiment.

Use `experiment.run_ab(current, candidate, inspect_output)` with completed synchronous
requests: one warmup each, then five balanced pairs. The callback inspects every output,
including warmups, and returns `{passed: bool, ...output evidence}`. Failures stay in the
record. Preserve snapshots/input/configuration identifiers in invocation metadata. Long-lived
services may make one callable a complete invocation; report that block granularity.

Report deployment delta separately from eager ratio and best-available isolated ratio.
A sample minimum does not establish statistical significance. Do not take the best repeat,
subtract independent medians, sum nested phases, or attribute relocation/overlap gains to
fusion without a separate controlled comparison. No automatic extra runs chase SOL gates.
