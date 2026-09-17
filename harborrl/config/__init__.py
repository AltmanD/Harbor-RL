"""Explicit configuration for the interactive Slime launcher transition."""
from __future__ import annotations

import os
from pathlib import Path
import re
from urllib.parse import urlsplit

import yaml

from harborrl.harnesses.identity import get_harness_descriptor

ROOT = Path(__file__).resolve().parents[2]
# Backend tuning remains explicit in the profile, using the launcher's names.
FIELDS = {
    'tasks': {'catalog'}, 'execution': {'backend'}, 'harness': {'name'},
    'model': {'provider', 'checkpoint', 'reference', 'args_file'},
    'training': {'backend', 'engine', 'algorithm', 'logprob_source', 'logprob_semantics'},
    'deployment': {'worker_urls', 'num_gpus', 'actor_gpus', 'rollout_gpus'},
    'output': {'root'},
}


def load_config(path, overrides=()):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('configuration must be a mapping')
    for override in overrides:
        key, sep, value = override.partition('=')
        parts = key.split('.')
        if not sep or len(parts) != 2 or parts[0] not in FIELDS or parts[1] not in FIELDS[parts[0]]:
            raise ValueError(f'unknown override: {key}')
        config.setdefault(parts[0], {})[parts[1]] = yaml.safe_load(value)
    if config.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    if set(config) - (set(FIELDS) | {'schema_version'}):
        raise ValueError('unknown configuration section')
    def expand(value):
        if isinstance(value, str):
            def replace(match):
                name = match[1]
                if not os.environ.get(name):
                    raise ValueError(f'missing declared environment variable: {name}')
                return os.environ[name]
            return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', replace, value)
        return value
    for section, fields in FIELDS.items():
        values = config.get(section)
        if not isinstance(values, dict) or set(values) != fields:
            raise ValueError(f'{section} requires exactly: {", ".join(sorted(fields))}')
        for key, value in values.items():
            value = expand(value)
            if not isinstance(value, (str, int)) or isinstance(value, bool) or value == '':
                raise ValueError(f'missing {section}.{key}')
            values[key] = value
    for section, key in [('tasks', 'catalog'), ('model', 'checkpoint'), ('model', 'reference'), ('output', 'root')]:
        value = Path(config[section][key]).expanduser()
        config[section][key] = str((path.parent / value).resolve() if not value.is_absolute() else value)
    for section, key, expected in [('execution','backend','lightrl_interactive'), ('model','provider','sglang'), ('training','backend','slime'), ('training','engine','megatron'), ('training','algorithm','grpo'), ('training','logprob_semantics','raw_model')]:
        if config[section][key] != expected:
            raise ValueError(f'unsupported {section}.{key}; currently requires {expected}')
    config['harness']['name'] = get_harness_descriptor(config['harness']['name'], capability='train').canonical_name
    for url in str(config['deployment']['worker_urls']).split(','):
        parsed = urlsplit(url.strip())
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('worker_urls must contain HTTP URLs without embedded credentials or query strings')
    d = config['deployment']
    for key in ('num_gpus', 'actor_gpus', 'rollout_gpus'):
        if type(d[key]) is not int or d[key] <= 0:
            raise ValueError(f'deployment.{key} must be a positive integer')
    if d['actor_gpus'] + d['rollout_gpus'] > d['num_gpus']:
        raise ValueError('actor and rollout GPUs exceed the node budget')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', str(config['model']['args_file'])):
        raise ValueError('model.args_file must name a bundled model preset')
    return config


def launch_plan(config):
    c = config
    env = {
        'DATASET': 'harbor_terminal', 'ALGO': 'grpo',
        'ROLLOUT_PROMPT_DATA': c['tasks']['catalog'],
        'HF_CKPT': c['model']['checkpoint'], 'REF_LOAD': c['model']['reference'],
        'MODEL_ARGS_FILE': c['model']['args_file'],
        'MODEL_TAG': c['model']['args_file'],
        'HARNESS_OPTION': get_harness_descriptor(c['harness']['name']).display_name,
        'WORKER_URLS': c['deployment']['worker_urls'],
        'NUM_GPUS': c['deployment']['num_gpus'], 'ACTOR_GPUS': c['deployment']['actor_gpus'],
        'ROLLOUT_GPUS': c['deployment']['rollout_gpus'],
        'TP_SIZE': c['deployment']['actor_gpus'],
        'ROLLOUT_NUM_GPUS_PER_ENGINE': c['deployment']['rollout_gpus'],
        'RUNS_ROOT': c['output']['root'], 'HARBOR_IR_ROOT': str(Path(c['output']['root']) / 'trajectories'),
        'HARBOR_LOGPROB_SOURCE': c['training']['logprob_source'],
        'HARBOR_LOGPROB_SEMANTICS': 'raw_model',
        'SLIME_ENTRYPOINT': str(ROOT / 'backends/slime/train.py'),
        'EXPLORE_INTRINSIC': '0', 'EXPLORE_AGENT57_LITE': '0', 'DAPO_OVERLONG_BUFFER_ENABLE': '0',
        'HARBORRL_SKIP_GLOBAL_CLEANUP': '1',
    }
    return {'config': c, 'environment': {k: str(v) for k,v in env.items()},
            'command': ['bash', str(ROOT / 'harborrl/platform/slime_train.sh')],
            'required_services': ['environment worker', 'local Ray and SGLang (launcher managed)'],
            'launcher': 'legacy Slime shell; unmigrated tuning retains backend defaults'}
