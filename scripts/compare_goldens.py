#!/usr/bin/env python3
"""Post-blind comparison, using golden cases and unchanged generated packages.

Run inside the GPU image, with checkout and run directory mounted read-only.
Each case runs in a fresh process; JSON and logs survive errors and timeouts.
This is a scoped comparison, not a replacement for the full acceptance suite.
"""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def load_cases(checkout, example):
    root = checkout / 'examples' / example
    sys.path[:0] = [str(root / 'golden'), str(checkout / 'kda/kda-skills/kda-kernel-scaffold/template')]
    spec = importlib.util.spec_from_file_location('comparison_cases', root / 'benchmark_cases.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generated(workspace):
    interfaces = sorted(workspace.glob('kda_kernels/*/interface.py'))
    if len(interfaces) != 1:
        raise ValueError('Expected one generated interface, found %s' % interfaces)
    path = interfaces[0]
    names = [n.name for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and not n.name.startswith(('_', 'set_'))]
    if len(names) != 1:
        raise ValueError('Ambiguous public operation: %s' % names)
    sys.path.insert(0, str(workspace))
    name = 'kda_kernels.' + path.parent.name
    api = importlib.import_module(name + '.interface')
    backends = importlib.import_module(name + '.backends')
    # Explicit kernel backend avoids an eager default disguising missing integration.
    spec_text = (path.parent / 'SPEC.md').read_text()
    import re
    match = re.search(r'^kernel_backend:\s*(\w+)', spec_text, re.M)
    backend = match.group(1) if match else backends.BACKEND_DEFAULT
    if backend in ('eager', 'auto') or backend.endswith('_fwd_only'):
        raise ValueError('Not a full generated kernel backend: ' + backend)
    fn = getattr(api, names[0])
    def call(*args, **kwargs):
        # Golden uses a boolean; generated AdaLN uses the equivalent enum.
        if 'silu' in kwargs:
            kwargs['act'] = 'silu' if kwargs.pop('silu') else 'none'
        if kwargs.get('recompute') and not backends.RECOMPUTE_AVAILABLE:
            raise ValueError('Generated package does not support recompute')
        return fn(*args, backend=backend, **kwargs)
    return call, {'module': name, 'function': names[0], 'backend': backend}


def worker(opts):
    import torch
    cases = load_cases(opts.checkout, opts.example)
    from _common import bench, verify
    row = next(r for r in cases.cases() if r['name'] == opts.case)
    result = dict(example=opts.example, case=opts.case, required=row.get('required', True),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, method='profiler',
                  warmup=3, iterations=10, rounds=3, seed=0, implementations={})
    save(opts.result, result)
    torch.manual_seed(0)
    args, kwargs = row['make']()
    result['inputs'] = [dict(shape=list(a.shape), stride=list(a.stride()), dtype=str(a.dtype))
                        if isinstance(a, torch.Tensor) else str(a) for a in args]
    result['kwargs'] = kwargs
    phases = row.get('phases', ['fwd', 'infer', 'bwd'])
    dtype = verify.lowest_precision(*(a.dtype for a in args if isinstance(a, torch.Tensor)))
    norm = lambda o: tuple(o) if isinstance(o, (tuple, list)) else (o,)
    wrt = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
    expected = norm(cases.reference_fn(*args, **kwargs))
    upstream = tuple(torch.randn_like(o) for o in expected)
    grad = lambda out: torch.autograd.grad(out, wrt, upstream, retain_graph=True)
    expected_grad = grad(expected) if 'bwd' in phases else ()
    reductions = row.get('grad_reductions', [1] * len(expected_grad))
    if len(reductions) != len(expected_grad) and 'bwd' in phases:
        raise ValueError('Gradient reduction metadata length mismatch')

    def compare(actual, ref, reduction=None):
        if len(actual) != len(ref):
            raise ValueError('Output/gradient count mismatch')
        checks = []
        for i, (a, e) in enumerate(zip(actual, ref)):
            precision = dtype if reduction is not None else verify.lowest_precision(a.dtype, e.dtype, dtype)
            atol, rtol = verify.TOLERANCES[precision]
            check = verify.compare(a, e, atol=atol, rtol=rtol,
                                   reduced_over=reduction[i] if reduction is not None else 1).as_dict()
            check['dtype_matches'] = a.dtype == e.dtype
            check['passed'] = check['passed'] and check['dtype_matches']
            checks.append(check)
        return checks

    functions = {'golden': cases.kernel_fn}
    try:
        functions['agent'], result['agent_entry'] = generated(opts.run / 'attempts' / opts.example / '1/workspace')
    except Exception:
        result['implementations']['agent'] = {'error': traceback.format_exc()}
    for name, fn in functions.items():
        record = result['implementations'][name] = {}
        try:
            actual = norm(fn(*args, **kwargs))
            record['fwd'] = compare(actual, expected)
            if 'bwd' in phases:
                record['bwd'] = compare(grad(actual), expected_grad, reductions)
            if 'infer' in phases:
                with torch.no_grad():
                    record['infer'] = compare(norm(fn(*args, **kwargs)), norm(cases.reference_fn(*args, **kwargs)))
            record['passed'] = all(c['passed'] for p in ('fwd', 'infer', 'bwd') for c in record.get(p, []))
            del actual
        except Exception:
            record['error'] = traceback.format_exc()
            record['passed'] = False
        save(opts.result, result)
    del expected, expected_grad
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if row.get('benchmark', True):
        for phase in phases:
            closures = {}
            for name, fn in functions.items():
                rec = result['implementations'][name]
                if not rec.get('passed'):
                    continue
                if phase == 'bwd':
                    out = norm(fn(*args, **kwargs))
                    closures[name] = lambda out=out: grad(out)
                elif phase == 'infer':
                    def infer(fn=fn):
                        with torch.no_grad():
                            return fn(*args, **kwargs)
                    closures[name] = infer
                else:
                    closures[name] = lambda fn=fn: fn(*args, **kwargs)
            samples = {name: [] for name in closures}
            for round_index in range(3):
                order = list(closures)
                if round_index % 2:
                    order.reverse()
                for name in order:
                    try:
                        samples[name].append(bench.bench_ms(closures[name], method='profiler', warmup=3, iters=10))
                    except Exception:
                        result['implementations'][name].setdefault('timing_errors', {})[phase] = traceback.format_exc()
            for name, times in samples.items():
                rec = result['implementations'][name]
                rec.setdefault('samples_ms', {})[phase] = times
                if len(times) == 3:
                    rec.setdefault('ms', {})[phase] = min(times)
            del closures
            save(opts.result, result)
    result['agent_over_golden'] = {}
    g = result['implementations'].get('golden', {}).get('ms', {})
    a = result['implementations'].get('agent', {}).get('ms', {})
    for phase in g.keys() & a.keys():
        result['agent_over_golden'][phase] = a[phase] / g[phase]
    result['finished'] = True
    save(opts.result, result)


def suite(opts):
    opts.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(started=time.time(), run=str(opts.run), cases=[], excluded=[], hashes={})
    # Metadata only: no credentials or native agent logs are copied.
    for root in (opts.checkout / 'examples', opts.run / 'attempts'):
        for f in root.rglob('*.py'):
            if 'golden' in f.parts or 'kda_kernels' in f.parts or f.name == 'benchmark_cases.py':
                manifest['hashes'][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    manifest['hashes'][str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(opts.output / 'manifest.json', manifest)

    examples = opts.examples.split(',') if opts.examples else sorted(p.name for p in (opts.checkout / 'examples').iterdir() if (p / 'benchmark_cases.py').exists())
    for example in examples:
        listing = subprocess.run([sys.executable, __file__, 'list', '--checkout', str(opts.checkout), '--example', example], capture_output=True, text=True, check=True)
        for case in json.loads(listing.stdout):
            if case.startswith('real_') or case.startswith('model_') or case.endswith('_recompute'):
                manifest['excluded'].append([example, case, 'large-scale or separate recompute coverage outside initial comparison'])
                continue
            stem = example + '--' + case
            path = opts.output / (stem + '.json')
            cmd = [sys.executable, __file__, 'worker', '--checkout', str(opts.checkout), '--run', str(opts.run), '--example', example, '--case', case, '--result', str(path)]
            print('START', stem, flush=True)
            with (opts.output / (stem + '.log')).open('w') as log:
                p = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
                try:
                    code = p.wait(timeout=opts.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    code = 124
                except BaseException:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    raise
            manifest['cases'].append(dict(example=example, case=case, exit_code=code, result=path.name))
            save(opts.output / 'manifest.json', manifest)
            print('END', stem, code, flush=True)
    manifest['ended'] = time.time()
    save(opts.output / 'manifest.json', manifest)


def summarize(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    rows = []
    for item in manifest['cases']:
        path = output / item['result']
        r = json.loads(path.read_text()) if path.exists() else {}
        im = r.get('implementations', {})
        rows.append(dict(example=item['example'], case=item['case'], exit_code=item['exit_code'],
                         finished=r.get('finished', False),
                         golden_passed=im.get('golden', {}).get('passed'),
                         agent_passed=im.get('agent', {}).get('passed'),
                         ratio=r.get('agent_over_golden', {}),
                         golden_ms=im.get('golden', {}).get('ms', {}),
                         agent_ms=im.get('agent', {}).get('ms', {})))
    by_example = {}
    for row in rows:
        by_example.setdefault(row['example'], []).append(row)
    summary = dict(cases=rows, excluded=manifest['excluded'], completed=bool(manifest.get('ended')), examples={})
    lines = ['# Golden comparison', '',
             'Same-device profiler comparison. Ratio = agent time / golden time; >1 means agent is slower.',
             'Correctness uses the unchanged golden eager reference and benchmark tolerances. This is scoped evidence, not full acceptance.', '',
             '| Example | Cases | Golden correct | Agent correct | Median fwd ratio | Median bwd ratio | Median infer ratio |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for example, group in by_example.items():
        ratios = {phase: [r['ratio'][phase] for r in group if phase in r['ratio']] for phase in ('fwd', 'bwd', 'infer')}
        medians = {p: statistics.median(v) if v else None for p, v in ratios.items()}
        g = sum(r['golden_passed'] is True for r in group)
        a = sum(r['agent_passed'] is True for r in group)
        summary['examples'][example] = dict(cases=len(group), golden_passed=g, agent_passed=a, median_ratios=medians)
        lines.append('| %s | %s | %s | %s | %s |' % (example, len(group), g, a, ' | '.join('%.2fx' % medians[p] if medians[p] else 'unavailable' for p in ('fwd', 'bwd', 'infer'))))
    lines += ['', '## Individual cases', '', '| Example / case | Exit | Golden / agent correct | fwd ratio | bwd ratio | infer ratio |', '|---|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append('| %s / %s | %s | %s / %s | %s |' % (r['example'], r['case'], r['exit_code'], r['golden_passed'], r['agent_passed'], ' | '.join('%.3fx' % r['ratio'][p] if p in r['ratio'] else 'unavailable' for p in ('fwd', 'bwd', 'infer'))))
    lines += ['', '## Excluded cases', ''] + ['- %s / %s: %s' % tuple(r) for r in manifest['excluded']]
    save(output / 'summary.json', summary)
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['suite', 'list', 'worker', 'summarize'])
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--run', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--example')
    p.add_argument('--examples')
    p.add_argument('--case')
    p.add_argument('--result', type=Path)
    p.add_argument('--timeout', type=int, default=240)
    opts = p.parse_args()
    if opts.mode == 'summarize':
        summarize(opts.output)
    elif opts.mode == 'list':
        print(json.dumps([r['name'] for r in load_cases(opts.checkout, opts.example).cases()]))
    elif opts.mode == 'worker':
        try:
            worker(opts)
        except Exception:
            record = json.loads(opts.result.read_text()) if opts.result.exists() else {}
            record['error'] = traceback.format_exc()
            save(opts.result, record)
            raise
    else:
        suite(opts)


if __name__ == '__main__':
    main()
