#!/usr/bin/env python3
"""Foreground, Docker-isolated KDA evaluations. Python 3.7+, no pip dependencies."""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from blind_agents import Adapter, configure

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / 'KDA-blind-runs'
LABEL = 'org.kda.blind-eval'
ACTIVE = ('starting', 'running')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def command(args, check=True, timeout=60):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'{args[0]} failed ({result.returncode}): {result.stderr[-2000:]}')
    return result


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Already locked: ' + str(path)) from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def duration(value):
    match = re.fullmatch(r'(\d+)([smh]?)', value)
    if not match or int(match[1]) == 0:
        raise argparse.ArgumentTypeError('Expected positive duration, e.g. 30m or 2h')
    return int(match[1]) * {'': 1, 's': 1, 'm': 60, 'h': 3600}[match[2]]


def gpu_list(value):
    result = value.split(',')
    if not all(x.isdigit() for x in result) or len(set(result)) != len(result):
        raise ValueError('GPU IDs must be distinct integers separated by commas')
    return result


def gpu_status():
    result = command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                      '--format=csv,noheader,nounits'])
    return [dict(zip(('index', 'uuid', 'name', 'memory_mib', 'utilization'),
                     [x.strip() for x in line.split(',')])) for line in result.stdout.splitlines()]


def base_container(config, home=None, gpu=None):
    args = ['--init', '--user', f'{os.getuid()}:{os.getgid()}', '--shm-size', '8g',
            '--workdir', '/workspace/task', '--env', 'HOME=/agent-home']
    if gpu is not None:
        args += ['--gpus', 'device=' + gpu]
    if config.get('runtime'):
        args += ['--mount', f'type=bind,src={config["runtime"]},dst=/opt/agent,readonly']
    if home:
        args += ['--mount', f'type=bind,src={home},dst=/agent-home']
    for key, value in Adapter(config).environment().items():
        args += ['--env', key + '=' + value]
    if config.get('auth_env'):
        # Docker inherits this variable; its value never appears in argv or the manifest.
        args += ['--env', config['auth_env']]
    return args


def doctor(config):
    image = command(['docker', 'image', 'inspect', config['image'], '--format', '{{.Id}}']).stdout.strip()
    config['image'] = image
    runtime = []
    if config.get('runtime'):
        runtime = ['--mount', f'type=bind,src={config["runtime"]},dst=/opt/agent,readonly']
    prefix = ['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
              *runtime, '--entrypoint', config['executable'], image]
    version = command(prefix + ['--version']).stdout.strip()
    help_args = ['exec', '--help'] if config['agent'] == 'codex' else ['--help']
    help_text = command(prefix + help_args).stdout
    Adapter(config).preflight(help_text)
    # Inspect the GPU interpreter without allocating kernels or touching another container.
    python = command(['docker', 'run', '--rm', '--entrypoint', 'python', image, '-c',
                      'import sys,torch,triton; print(sys.version); print(torch.__version__,triton.__version__)']).stdout
    return {'image': image, 'cli_version': version, 'python': python, 'gpus': gpu_status(),
            'authentication': 'configured; not authenticated by doctor',
            'target_validation': 'not yet smoke-tested'}


def safe_copy(src, dst):
    # Dereference only links inside the source tree; never import an outside answer.
    src = src.resolve()
    for path in src.rglob('*'):
        if any(part.startswith('._') or part == '__MACOSX' for part in path.relative_to(src).parts):
            continue
        if path.is_symlink() and src not in path.resolve().parents and path.resolve() != src:
            raise ValueError('Input symlink escapes fixture: ' + str(path))
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('.git', '__pycache__', '.DS_Store', '._*', '__MACOSX'))


def remove_metadata(root):
    for path in sorted(root.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.name.startswith('._') or path.name in ('.DS_Store', '__MACOSX'):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def new_batch(config, examples, output, smoke=False):
    names = sorted(p.name for p in (ROOT / 'examples').iterdir()
                   if (p / 'TASK.md').is_file() and (p / 'user_repo').is_dir())
    if examples:
        chosen = examples.split(',')
        if len(set(chosen)) != len(chosen) or set(chosen) - set(names):
            raise ValueError('Unknown or duplicate examples: ' + examples)
        names = sorted(chosen)
    batch = output / (datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    batch.mkdir(parents=True)
    snapshots = batch / 'snapshots'
    snapshots.mkdir()
    for index, name in enumerate(names, 1):
        print(f'Freezing input {index}/{len(names)}: {name}', flush=True)
        dst = snapshots / name
        safe_copy(ROOT / 'examples' / name / 'user_repo', dst)
        shutil.copyfile(ROOT / 'examples' / name / 'TASK.md', dst / 'TASK.md')
        command(['bash', str(ROOT / 'kda/install.sh'), str(dst)], timeout=180)
        remove_metadata(dst)
    manifest = {'id': batch.name, 'created_at': now(), 'config': config, 'smoke': smoke,
                'commit': command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).stdout.strip(),
                'examples': names, 'input_hashes': hashes(snapshots),
                'runner_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), Path(__file__).with_name('blind_agents.py'))},
                'verdict_source': 'agent report; no independent acceptance'}
    write_json(batch / 'manifest.json', manifest)
    write_json(batch / 'state.json', {'attempts': [], 'updated_at': now()})
    return batch


def report_verdict(work):
    reports = []
    for path in sorted((work / 'kda_kernels').glob('*/report.json')):
        if work.resolve() not in path.resolve().parents:
            reports.append({'path': str(path.relative_to(work)), 'agent_verdict': 'unknown'})
            continue
        try:
            verification = read_json(path).get('verification', {})
            decision = verification.get('verdict', 'unknown') if isinstance(verification, dict) else 'unknown'
            if decision not in ('pass', 'tune', 'fail', 'incomplete'):
                decision = 'unknown'
            reports.append({'path': str(path.relative_to(work)), 'agent_verdict': decision})
        except (ValueError, OSError, AttributeError):
            reports.append({'path': str(path.relative_to(work)), 'agent_verdict': 'unknown'})
    return reports


# Lives outside agent-controlled workspace. timeout also applies if the host scheduler dies.
SUPERVISOR = '''import json,os,signal,subprocess,sys
request=json.load(open('/run/kda-request.json'))
p=None
def interrupted(*args):
 raise InterruptedError()
signal.signal(signal.SIGTERM,interrupted)
try:
 p=subprocess.Popen(request['argv'],start_new_session=True)
 try: code=p.wait(timeout=request['timeout'])
 except subprocess.TimeoutExpired: code=124
except InterruptedError: code=143
except Exception as exc:
 print(str(exc),file=sys.stderr); code=125
finally:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 if p is not None and p.poll() is None:
  try: os.killpg(p.pid,signal.SIGTERM)
  except ProcessLookupError: pass
  try: p.wait(timeout=14)
  except subprocess.TimeoutExpired:
   try: os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError: pass
   p.wait()
sys.exit(code)
'''


def inspect_owned(name, batch_id):
    result = command(['docker', 'inspect', name], check=False)
    if result.returncode:
        if 'no such' in result.stderr.lower():
            return None
        raise RuntimeError('Cannot inspect container: ' + result.stderr[-1000:])
    info = json.loads(result.stdout)[0]
    if info['Config'].get('Labels', {}).get(LABEL) != batch_id:
        raise RuntimeError('Refusing to operate on foreign container: ' + name)
    return info


def remove_owned(name, batch_id):
    info = inspect_owned(name, batch_id)
    if info:
        if info['State']['Running']:
            command(['docker', 'stop', '--time', '15', name], timeout=25)
        info = inspect_owned(name, batch_id)
        command(['docker', 'rm', name])
    return info


def reconcile(batch, state):
    for attempt in state['attempts']:
        if attempt.get('cleanup_error'):
            remove_owned(attempt['container'], batch.name)
            shutil.rmtree(attempt['private_dir'], ignore_errors=True)
            del attempt['cleanup_error']
        if attempt['status'] not in ACTIVE:
            continue
        info = inspect_owned(attempt['container'], batch.name)
        # Caller holds the batch lock, so these containers have no live scheduler.
        if info:
            if info['State']['Running']:
                command(['docker', 'stop', '--time', '15', attempt['container']], timeout=25)
            info = inspect_owned(attempt['container'], batch.name)
            logs = command(['docker', 'logs', attempt['container']], check=False)
            directory = batch / attempt['directory']
            (directory / 'recovered.stdout.log').write_text(logs.stdout)
            (directory / 'recovered.stderr.log').write_text(logs.stderr)
            attempt['exit_code'] = info['State']['ExitCode']
            remove_owned(attempt['container'], batch.name)
        attempt.update(status='interrupted', ended_at=now())
        if attempt.get('private_dir'):
            shutil.rmtree(attempt['private_dir'], ignore_errors=True)
    write_json(batch / 'state.json', state)


def summary(batch, state):
    write_json(batch / 'summary.json', state)
    lines = ['# Blind evaluation', '', 'Verdicts are agent-reported; not independent acceptance.', '',
             '| Example | Attempt | GPU | Run status | Agent verdicts |', '|---|---:|---|---|---|']
    for a in state['attempts']:
        verdicts = ', '.join(x['agent_verdict'] for x in a.get('reports', [])) or 'unknown'
        lines.append(f'| {a["example"]} | {a["number"]} | {a["gpu"]} | {a["status"]} | {verdicts} |')
    (batch / 'summary.md').write_text('\n'.join(lines) + '\n')


def display(batch, state):
    print(f'\n{batch.name}  {now()}', flush=True)
    manifest = read_json(batch / 'manifest.json')
    attempted = {a['example'] for a in state['attempts']}
    print('Pending:', ', '.join(n for n in manifest['examples'] if n not in attempted) or 'none', flush=True)
    for a in state['attempts']:
        print(f'{a["example"]} #{a["number"]} {a.get("agent", "")}:{a.get("model", "")} GPU={a["gpu"]} {a["status"]} '
              f'elapsed={a.get("elapsed_seconds", 0):.0f}s last={a.get("last_event_at", "-")} '
              f'{a.get("activity", "")[:100]} container={a["container"]}', flush=True)
        if a.get('reported_usage') or a.get('reported_cost_usd') is not None:
            print('  Reported usage:', a.get('reported_usage', 'unknown'),
                  'reported cost USD:', a.get('reported_cost_usd', 'unknown'), flush=True)


def schedule(batch, gpus, retry=None, smoke=False):
    with lock(batch / '.lock'), contextlib.ExitStack() as stack:
        for gpu in sorted(gpus):
            stack.enter_context(lock(Path('/tmp') / f'kda-blind-{os.getuid()}-gpu-{gpu}.lock'))
        manifest = read_json(batch / 'manifest.json')
        smoke = manifest.get('smoke', smoke)
        if hashes(batch / 'snapshots') != manifest['input_hashes']:
            raise ValueError('Frozen input snapshot changed; create a new batch')
        config = manifest['config']
        adapter = Adapter(config)
        state = read_json(batch / 'state.json')
        reconcile(batch, state)
        if retry and retry not in manifest['examples']:
            raise ValueError('Example not in original batch: ' + retry)
        attempted = {a['example'] for a in state['attempts']}
        pending = [retry] if retry else [n for n in manifest['examples'] if n not in attempted]
        stopfile = batch / 'STOP'
        if stopfile.exists():
            stopfile.unlink()
        stop = threading.Event()
        events = queue.Queue()
        workers = {}
        available = list(gpus)
        gpu_map = {g['index']: g for g in gpu_status()}
        if set(gpus) - gpu_map.keys():
            raise ValueError('Unknown GPU IDs')
        old = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            old[sig] = signal.signal(sig, lambda *_: stop.set())

        def worker(a, private):
            directory = batch / a['directory']
            work = directory / 'workspace'
            name = a['container']
            started = time.monotonic()
            terminal = {'status': 'failed'}
            log_process = None
            try:
                safe_copy(batch / 'snapshots' / a['example'], work)
                # Installer links are internal. safe_copy has materialized them for each attempt.
                home = private / 'home'
                adapter.prepare(home)
                if smoke:
                    prompt = ('Use the shell tool to run python importing torch and triton, print their '
                              'versions, CUDA device count and GPU name, and compute '
                              'torch.arange(16,device="cuda").sum().item(). Write the result to '
                              'SMOKE.json in this workspace with keys device_count (integer), sum (number), '
                              'torch_version, triton_version, gpu_name. Create a CUDA torch.nn.Parameter, '
                              'construct torch.optim.AdamW, backpropagate its squared sum and run one optimizer '
                              'step; include optimizer_step=true only if this succeeds. '
                              'Also check that /workspace/KDA/examples '
                              'and /var/run/docker.sock do not exist; include isolated (boolean). '
                              'Do not optimize or read TASK.md. Then finish.')
                else:
                    prompt = ('Read .agents/skills/kda-kernels/SKILL.md and follow KDA for the task below. '
                              'Use python in this container for GPU checks. Main and worker model must be '
                              + config['model'] + '. Use sequential stages and workers; if no fresh-context '
                              'worker exists record independence: same_context. Preserve eager fallback. '
                              'Record actual evidence and unavailable checks. Do not search for withheld '
                              'solutions online or outside this workspace.\n\n' + (work / 'TASK.md').read_text())
                (directory / 'prompt.txt').write_text(prompt)
                request = private / 'request.json'
                write_json(request, {'argv': adapter.build_command(prompt),
                                     'timeout': min(config['timeout'], 180) if smoke else config['timeout']})
                args = ['docker', 'create', '--name', name, '--label', LABEL + '=' + batch.name,
                        *base_container(config, home, a['gpu']),
                        '--mount', f'type=bind,src={work},dst=/workspace/task',
                        '--mount', f'type=bind,src={request},dst=/run/kda-request.json,readonly',
                        '--entrypoint', 'python', config['image'], '-c', SUPERVISOR]
                command(args)
                if stop.is_set():
                    terminal['status'] = 'stopped'
                    return
                command(['docker', 'start', name])
                events.put((a, {'status': 'running'}))
                with (directory / 'trajectory.jsonl').open('wb') as raw, (directory / 'stderr.log').open('w') as err:
                    log_process = subprocess.Popen(['docker', 'logs', '--follow', name],
                                                   stdout=subprocess.PIPE, stderr=err)
                    def consume():
                        for line in log_process.stdout:
                            if isinstance(line, str):
                                line = line.encode('utf-8')
                            raw.write(line)
                            raw.flush()
                            try:
                                parsed = adapter.parse_event(json.loads(line.decode('utf-8', errors='replace')))
                            except (ValueError, TypeError):
                                parsed = {'kind': 'unparsed'}
                            parsed['time'] = now()
                            events.put((a, {'event': parsed}))
                    reader = threading.Thread(target=consume, daemon=True)
                    reader.start()
                    while True:
                        info = inspect_owned(name, batch.name)
                        if not info or not info['State']['Running']:
                            code = info['State']['ExitCode'] if info else None
                            terminal.update(exit_code=code, status='completed' if code == 0 else
                                            'timed_out' if code == 124 else 'failed')
                            break
                        if stop.is_set() or time.monotonic() - started > (180 if smoke else config['timeout']) + 20:
                            terminal['status'] = 'stopped' if stop.is_set() else 'timed_out'
                            command(['docker', 'stop', '--time', '15', name], timeout=25)
                            info = inspect_owned(name, batch.name)
                            terminal['exit_code'] = info['State']['ExitCode'] if info else None
                            break
                        stop.wait(1)
                    log_process.wait(timeout=15)
                    reader.join(timeout=5)
                    log_process.stdout.close()
                terminal['reports'] = report_verdict(work)
                if smoke:
                    try:
                        evidence = read_json(work / 'SMOKE.json')
                        terminal['smoke_passed'] = (evidence.get('device_count') == 1 and
                                                    evidence.get('sum') == 120 and evidence.get('isolated') is True
                                                    and evidence.get('optimizer_step') is True
                                                    and bool(evidence.get('torch_version'))
                                                    and bool(evidence.get('triton_version')))
                    except (OSError, ValueError, AttributeError):
                        terminal['smoke_passed'] = False
                    if not terminal['smoke_passed'] and terminal['status'] == 'completed':
                        terminal['status'] = 'failed'
                        terminal['error'] = 'Smoke evidence missing or invalid'
            except Exception as exc:
                terminal['error'] = str(exc)
            finally:
                try:
                    remove_owned(name, batch.name)
                except Exception as exc:
                    terminal['cleanup_error'] = str(exc)
                else:
                    shutil.rmtree(private, ignore_errors=True)
                if log_process and log_process.poll() is None:
                    log_process.terminate()
                terminal.update(ended_at=now(), elapsed_seconds=time.monotonic() - started)
                events.put((a, {'done': terminal}))

        try:
            last_display = 0
            while pending or workers:
                if stopfile.exists():
                    stop.set()
                while pending and available and not stop.is_set():
                    example, gpu = pending.pop(0), available.pop(0)
                    number = 1 + sum(a['example'] == example for a in state['attempts'])
                    rel = f'attempts/{example}/{number}'
                    (batch / rel).mkdir(parents=True)
                    private = Path(tempfile.mkdtemp(prefix='kda-blind-'))
                    a = {'example': example, 'number': number, 'gpu': gpu, 'gpu_uuid': gpu_map[gpu]['uuid'],
                         'agent': config['agent'], 'model': config['model'], 'model_observation': 'unknown',
                         'subagent_model_observation': 'unknown',
                         'container': f'kda-{batch.name}-{example[:24]}-{number}',
                         'directory': rel, 'private_dir': str(private), 'status': 'starting',
                         'started_at': now()}
                    state['attempts'].append(a)
                    write_json(batch / 'state.json', state)
                    thread = threading.Thread(target=worker, args=(a, private))
                    workers[gpu] = thread
                    thread.start()
                try:
                    a, update = events.get(timeout=0.2)
                    if 'event' in update:
                        event = update['event']
                        with (batch / a['directory'] / 'events.jsonl').open('a') as stream:
                            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                        a['last_event_at'] = event['time']
                        a['activity'] = event.get('tool') or event.get('message') or event['kind']
                        if event.get('session_id'):
                            a['session_id'] = event['session_id']
                        if event.get('usage'):
                            a['reported_usage'] = event['usage']
                        if 'total_cost_usd' in event:
                            a['reported_cost_usd'] = event['total_cost_usd']
                        if event.get('is_error') is True:
                            a['agent_error'] = True
                        if event.get('model'):
                            a.setdefault('observed_models', [])
                            if event['model'] not in a['observed_models']:
                                a['observed_models'].append(event['model'])
                            if event['model'] == config['model']:
                                a['model_observation'] = 'exact_id_seen; worker identities unconfirmed'
                            elif isinstance(event['model'], str) and ' ' in event['model']:
                                a['model_observation'] = 'display_label_seen; exact_id_unconfirmed'
                            else:
                                a['model_observation'] = 'different_model_id; inspect trajectory'
                    elif 'done' in update:
                        a.update(update['done'])
                        if a.get('agent_error') and a['status'] == 'completed':
                            a['status'] = 'failed'
                            a['error'] = 'Agent emitted an error result'
                        if a.get('cleanup_error'):
                            stop.set()  # Never reuse a GPU while its old container might still run.
                        workers.pop(a['gpu']).join()
                        available.append(a['gpu'])
                    else:
                        a.update(update)
                except queue.Empty:
                    pass
                if time.monotonic() - last_display >= 5:
                    for a in state['attempts']:
                        if a['status'] in ACTIVE:
                            a['elapsed_seconds'] = (datetime.datetime.now(datetime.timezone.utc) -
                                                    datetime.datetime.fromisoformat(a['started_at'])).total_seconds()
                    try:
                        sample = {'time': now(), 'gpus': gpu_status()}
                        with (batch / 'gpu.jsonl').open('a') as stream:
                            stream.write(json.dumps(sample) + '\n')
                        print('GPU:', ' | '.join(f'{g["index"]}: {g["memory_mib"]}MiB {g["utilization"]}%'
                                               for g in sample['gpus'] if g['index'] in gpus), flush=True)
                    except Exception as exc:
                        print('GPU sampling unavailable:', exc, flush=True)
                    display(batch, state)
                    last_display = time.monotonic()
                state['updated_at'] = now()
                write_json(batch / 'state.json', state)
                if stop.is_set() and not workers:
                    break
        finally:
            stop.set()
            for thread in workers.values():
                thread.join()
            while not events.empty():
                a, update = events.get()
                if 'done' in update:
                    a.update(update['done'])
            write_json(batch / 'state.json', state)
            summary(batch, state)
            for sig, handler in old.items():
                signal.signal(sig, handler)
        display(batch, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('run', 'smoke', 'doctor'):
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path)
        for key in ('agent', 'provider', 'model', 'effort', 'image', 'runtime', 'executable', 'auth-file', 'auth-env'):
            p.add_argument('--' + key)
        p.add_argument('--timeout', type=duration)
        p.add_argument('--gpus', required=name != 'doctor')
        p.add_argument('--examples')
        p.add_argument('--output-root', type=Path, default=OUTPUT)
    for name in ('status', 'logs', 'stop', 'resume', 'retry'):
        p = sub.add_parser(name)
        p.add_argument('run_id')
        p.add_argument('--output-root', type=Path, default=OUTPUT)
        if name == 'status':
            p.add_argument('--watch', action='store_true')
        if name == 'logs':
            p.add_argument('--example', required=True)
            p.add_argument('--follow', action='store_true')
        if name in ('resume', 'retry'):
            p.add_argument('--gpus', required=True)
        if name == 'retry':
            p.add_argument('--example', required=True)
    args = parser.parse_args()
    if args.action in ('run', 'doctor', 'smoke'):
        config = read_json(args.config) if args.config else {}
        if not isinstance(config, dict):
            raise ValueError('Configuration must be a JSON object')
        keys = ('agent', 'provider', 'model', 'effort', 'image', 'runtime', 'executable', 'auth_file', 'auth_env', 'timeout')
        unknown = set(config) - set(keys)
        if unknown:
            raise ValueError('Unknown config keys: ' + ', '.join(sorted(unknown)))
        config.update({k: getattr(args, k) for k in keys if getattr(args, k) is not None})
        config = configure(config)
        diagnostics = doctor(config)
        print(json.dumps(diagnostics, indent=2), flush=True)
        if args.action == 'doctor':
            return
        gpus = gpu_list(args.gpus)
        if set(gpus) - {x['index'] for x in diagnostics['gpus']}:
            raise ValueError('Unknown GPU IDs')
        if args.action == 'run':
            preflight = new_batch(config, 'fused_residual_rmsnorm', args.output_root.resolve() / 'preflight', smoke=True)
            print('PREFLIGHT=' + str(preflight), flush=True)
            smoke_state = schedule(preflight, gpus[:1], smoke=True)
            if not all(a.get('smoke_passed') and a['status'] == 'completed' for a in smoke_state['attempts']):
                raise RuntimeError('Model/GPU smoke failed; no evaluation examples were started')
            diagnostics['target_validation'] = 'smoke passed; see ' + str(preflight)
        batch = new_batch(config, args.examples or ('fused_residual_rmsnorm' if args.action == 'smoke' else None),
                          args.output_root.resolve(), smoke=args.action == 'smoke')
        write_json(batch / 'doctor.json', diagnostics)
        print('RUN_ID=' + batch.name, flush=True)
        state = schedule(batch, gpus, smoke=args.action == 'smoke')
        if any(a['status'] != 'completed' or a.get('cleanup_error') for a in state['attempts']):
            raise SystemExit(1)
        return
    batch = Path(args.run_id)
    if not batch.is_absolute():
        batch = args.output_root / batch
    batch = batch.resolve(strict=True)
    if args.action == 'stop':
        (batch / 'STOP').touch()
        # A live scheduler observes STOP. If it has died, take ownership and clean up.
        try:
            with lock(batch / '.lock'):
                state = read_json(batch / 'state.json')
                reconcile(batch, state)
                summary(batch, state)
        except RuntimeError as exc:
            if not str(exc).startswith('Already locked:'):
                raise
        print('Stop requested:', batch.name)
    elif args.action in ('resume', 'retry'):
        configure(read_json(batch / 'manifest.json')['config'])
        previous_count = len(read_json(batch / 'state.json')['attempts'])
        state = schedule(batch, gpu_list(args.gpus), retry=getattr(args, 'example', None))
        if any(a['status'] != 'completed' or a.get('cleanup_error') for a in state['attempts'][previous_count:]):
            raise SystemExit(1)
    elif args.action == 'status':
        while True:
            state = read_json(batch / 'state.json')
            try:
                with lock(batch / '.lock'):
                    reconcile(batch, state)
                    summary(batch, state)
            except RuntimeError as exc:
                if not str(exc).startswith('Already locked:'):
                    raise
            display(batch, state)
            print('Agent/model:', read_json(batch / 'manifest.json')['config']['agent'],
                  read_json(batch / 'manifest.json')['config']['model'])
            print('GPU:', gpu_status())
            if not args.watch:
                break
            time.sleep(5)
    elif args.action == 'logs':
        attempts = [a for a in read_json(batch / 'state.json')['attempts'] if a['example'] == args.example]
        if not attempts:
            raise ValueError('No attempt for ' + args.example)
        attempt = attempts[-1]
        path = batch / attempt['directory'] / 'trajectory.jsonl'
        position = 0
        while True:
            if path.exists():
                with path.open(errors='replace') as stream:
                    stream.seek(position)
                    sys.stdout.write(stream.read())
                    sys.stdout.flush()
                    position = stream.tell()
            current = next(a for a in read_json(batch / 'state.json')['attempts']
                           if a['directory'] == attempt['directory'])
            if not args.follow or current['status'] not in ACTIVE:
                break
            time.sleep(0.5)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print('error:', exc, file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print('Interrupted.', file=sys.stderr)
        sys.exit(130)
