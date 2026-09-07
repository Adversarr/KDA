"""Small, standard-library-only adapters for headless coding agents."""
import json
import math
import os
import shutil
from pathlib import Path


class Adapter:
    def __init__(self, config):
        self.config = config
        self.name = config['agent']

    def build_command(self, prompt):
        c = self.config
        exe = c['executable']
        model = c['model']
        effort = c.get('effort')
        if self.name == 'cursor':
            return [exe, '--print', '--force', '--trust', '--sandbox', 'disabled',
                    '--model', model, '--output-format', 'stream-json', prompt]
        if self.name == 'pi':
            args = [exe, '--print', '--mode', 'json', '--no-extensions',
                    '--provider', c['provider'], '--model', model]
            return args + (['--thinking', effort] if effort else []) + [prompt]
        if self.name == 'claude':
            args = [exe, '-p', '--verbose', '--output-format', 'stream-json',
                    '--dangerously-skip-permissions', '--model', model]
            return args + (['--effort', effort] if effort else []) + [prompt]
        args = [exe, 'exec', '--json', '--skip-git-repo-check',
                '--dangerously-bypass-approvals-and-sandbox', '-m', model]
        if effort:
            args += ['-c', 'model_reasoning_effort=' + json.dumps(effort)]
        return args + [prompt]

    def preflight(self, help_text):
        required = {
            'cursor': ['--print', '--model', '--output-format', '--sandbox', '--trust'],
            'pi': ['--print', '--mode', '--provider', '--model', '--no-extensions'],
            'claude': ['--print', '--output-format', '--model', '--dangerously-skip-permissions'],
            'codex': ['--json', '--model', '--dangerously-bypass-approvals-and-sandbox'],
        }[self.name]
        if self.config.get('effort'):
            required += [{'pi': '--thinking', 'claude': '--effort', 'codex': '--config'}[self.name]]
        missing = [arg for arg in required if arg not in help_text]
        if missing:
            raise ValueError('Installed CLI lacks required options: ' + ', '.join(missing))

    def prepare(self, home):
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        model = self.config['model']
        configs = {
            'cursor': ('.cursor/cli-config.json', {'version': 1, 'exploreSubagentModel': model,
                       'network': {'useHttp1ForAgent': True}}),
            'pi': ('.pi/agent/settings.json', {'defaultProvider': self.config.get('provider'),
                   'defaultModel': model}),
            'claude': ('.claude/settings.json', {'model': model}),
        }
        if self.name in configs:
            rel, data = configs[self.name]
            path = home / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        # Never inherit global histories, extensions, MCPs, or user configuration.
        targets = {'cursor': '.config/cursor/auth.json', 'pi': '.pi/agent/auth.json',
                   'claude': '.claude/.credentials.json', 'codex': '.codex/auth.json'}
        source = self.config.get('auth_file')
        if source:
            dst = home / targets[self.name]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dst)
            dst.chmod(0o600)

    def environment(self):
        env = {'HOME': '/agent-home', 'PYTHONUNBUFFERED': '1', 'MAX_JOBS': '8',
               'CUDA_VISIBLE_DEVICES': '0', 'USER': 'kda', 'LOGNAME': 'kda',
               'TORCHINDUCTOR_CACHE_DIR': '/agent-home/.cache/torchinductor',
               'TRITON_CACHE_DIR': '/agent-home/.cache/triton'}
        if self.name == 'claude':
            env['CLAUDE_CODE_SUBAGENT_MODEL'] = self.config['model']
        return env

    def parse_event(self, event):
        if not isinstance(event, dict):
            return {'kind': 'unknown'}
        kind = event.get('type', 'unknown')
        result = {'kind': kind}
        for key in ('session_id', 'model', 'usage', 'total_cost_usd', 'subtype', 'is_error'):
            if key in event:
                result[key] = event[key]
        if event.get('thread_id'):
            result['session_id'] = event['thread_id']
        item = event.get('item') or event.get('message') or {}
        if isinstance(item, dict):
            for key in ('model', 'usage'):
                if key in item:
                    result[key] = item[key]
            text = item.get('text') or item.get('command')
            content = item.get('content', [])
            if isinstance(content, list):
                text = text or ' '.join(str(x.get('text') or x.get('name') or '')
                                       for x in content if isinstance(x, dict))
            if text:
                result['message'] = str(text)[:500]
        if 'toolName' in event:
            result['tool'] = event['toolName']
        if 'tool_call' in event and isinstance(event['tool_call'], dict):
            result['tool'] = ','.join(k for k in event['tool_call'] if k.endswith('ToolCall'))
        if isinstance(event.get('result'), str):
            result['message'] = event['result'][:500]
        return result

    def cleanup(self, home):
        shutil.rmtree(home, ignore_errors=True)


def configure(values):
    if not isinstance(values, dict):
        raise ValueError('Configuration must be a JSON object')
    config = dict(values)
    for key, value in config.items():
        if key != 'timeout' and (not isinstance(value, str) or not value.strip()):
            raise ValueError('Config ' + key + ' must be a nonempty string')
    name = config.setdefault('agent', 'cursor')
    if name not in ('cursor', 'pi', 'claude', 'codex'):
        raise ValueError('Unknown agent: ' + name)
    if not config.get('model'):
        raise ValueError('--model is required for ' + name)
    if name == 'pi' and not config.get('provider'):
        raise ValueError('--provider is required for pi')
    if config.get('provider') and name != 'pi':
        raise ValueError('This version supports --provider only for pi')
    if config.get('effort') and name == 'cursor':
        raise ValueError('For Cursor, specify effort in its exact model ID')
    if config.get('runtime'):
        config['runtime'] = str(Path(config['runtime']).expanduser().resolve(strict=True))
        if not Path(config['runtime']).is_dir():
            raise ValueError('runtime must be a directory')
    elif name == 'cursor' and not config.get('executable'):
        binary = Path.home() / '.local/bin/cursor-agent'
        if binary.exists():
            config['runtime'] = str(binary.resolve().parent)
    config.setdefault('executable', '/opt/agent/cursor-agent' if name == 'cursor' and config.get('runtime')
                      else {'cursor': 'cursor-agent', 'pi': 'pi', 'claude': 'claude', 'codex': 'codex'}[name])
    if not config.get('auth_file') and not config.get('auth_env'):
        default = {'cursor': '.config/cursor/auth.json', 'pi': '.pi/agent/auth.json',
                   'claude': '.claude/.credentials.json', 'codex': '.codex/auth.json'}[name]
        source = Path.home() / default
        if source.is_file():
            config['auth_file'] = str(source)
    if config.get('auth_file'):
        config['auth_file'] = str(Path(config['auth_file']).expanduser().resolve(strict=True))
        if not Path(config['auth_file']).is_file():
            raise ValueError('auth_file must be a file')
    if config.get('auth_env') and config['auth_env'] not in os.environ:
        raise ValueError('Missing credential environment variable: ' + config['auth_env'])
    if not config.get('auth_file') and not config.get('auth_env'):
        raise ValueError('Supply auth_file or auth_env; credentials are never auto-installed')
    config.setdefault('image', 'kda:dev')
    config.setdefault('timeout', 7200)
    if (isinstance(config['timeout'], bool) or not isinstance(config['timeout'], (int, float))
            or not math.isfinite(config['timeout']) or config['timeout'] <= 0):
        raise ValueError('Config timeout must be a positive number of seconds')
    for key in ('runtime', 'auth_file'):
        if config.get(key) and ',' in config[key]:
            raise ValueError('Docker mount paths cannot contain commas: ' + key)
    return config
