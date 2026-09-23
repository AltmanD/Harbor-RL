"""Strict native schema 2; local paths resolve relative to the config file."""
import os
import json
from pathlib import Path
from urllib.parse import urlsplit
import sys
import re

from harborrl.rollout.harbor_job.bindings import claude_config
from harborrl.trajectories.native import RewardProfile

FIELDS = {
    'execution': {'backend'}, 'tasks': {'catalog'}, 'harness': {'name', 'profile'},
    'harbor': {'python', 'version', 'workers'},
    'gateway': {'host', 'port', 'advertised_url', 'tokenizer_digest', 'template_digest', 'audited_raw_logprobs'},
    'model': {'checkpoint', 'reference', 'args_file'},
    'training': {'num_rollout', 'learning_rate', 'save_interval', 'backend_contract'},
    'sampling': {'group_size', 'groups_per_batch', 'max_attempts', 'max_tokens', 'max_context'},
    'deployment': {'layout', 'num_gpus', 'actor_gpus', 'rollout_gpus', 'actor_tensor_parallel_size', 'rollout_gpus_per_engine'},
    'output': {'root'},
}


def validate(config, path):
    from harborrl.config import ROOT
    from harborrl.trajectories.native import read_json
    if set(config) != set(FIELDS) | {'schema_version'} or type(config['schema_version']) is not int or config['schema_version'] != 2:
        raise ValueError('native schema requires exactly the documented schema 2 sections')
    config['training'].setdefault('backend_contract', 'slime-legacy')
    for section, fields in FIELDS.items():
        if not isinstance(config[section], dict) or set(config[section]) != fields:
            raise ValueError(f'{section} requires exactly {sorted(fields)}')
    if config['execution']['backend'] != 'harbor_job' or config['harness']['name'] != 'claude_code' or config['harbor']['version'] != '0.23.0':
        raise ValueError('native requires harbor_job / claude_code / Harbor 0.23.0')
    for section, key in [('tasks','catalog'), ('harness','profile'), ('harbor','python'),
                         ('model','checkpoint'), ('model','reference'), ('output','root')]:
        value = config[section][key]
        if not isinstance(value, str) or not value:
            raise ValueError(f'{section}.{key} must be a path')
        expanded = Path(value).expanduser()
        absolute = expanded if expanded.is_absolute() else Path(path).parent / expanded
        config[section][key] = str(Path(os.path.abspath(absolute)))
    for section, keys in [('sampling', FIELDS['sampling']), ('deployment', FIELDS['deployment'] - {'layout'}),
                          ('training', {'num_rollout','save_interval'}), ('gateway', {'port'})]:
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] <= 0:
                raise ValueError(f'{section}.{key} must be a positive integer')
    from harborrl.trajectories.native import finite
    from harborrl.backends.slime_v032.versions import CONFIG_CONTRACTS
    if config['training']['backend_contract'] not in CONFIG_CONTRACTS:
        raise ValueError(f"training.backend_contract must be one of {sorted(CONFIG_CONTRACTS)}")
    if not finite(config['training']['learning_rate']) or config['training']['learning_rate'] <= 0:
        raise ValueError('positive finite learning_rate required')
    g, d, s = config['gateway'], config['deployment'], config['sampling']
    if type(g['audited_raw_logprobs']) is not bool or not 0 < g['port'] < 65536:
        raise ValueError('invalid gateway audit flag or port')
    for key in ('host','tokenizer_digest','template_digest'):
        if not isinstance(g[key], str) or not g[key].strip():
            raise ValueError(f'gateway.{key} required')
    advertised = urlsplit(g['advertised_url'])
    if (advertised.scheme not in ('http', 'https') or not advertised.hostname
            or advertised.username or advertised.password or advertised.query or advertised.fragment
            or advertised.path not in ('', '/') or advertised.port != g['port']):
        raise ValueError('advertised gateway must be a plain origin and match listener port')
    if s['group_size'] < 2 or s['max_tokens'] >= s['max_context']:
        raise ValueError('invalid group or context budget')
    if d['layout'] not in ('split','colocate'):
        raise ValueError('layout must be split or colocate')
    if d['layout'] == 'split' and d['actor_gpus'] + d['rollout_gpus'] > d['num_gpus']:
        raise ValueError('GPU budget exceeded')
    if d['layout'] == 'colocate' and not d['actor_gpus'] == d['rollout_gpus'] == d['num_gpus']:
        raise ValueError('colocate GPU counts must match')
    if d['actor_gpus'] % d['actor_tensor_parallel_size'] or d['rollout_gpus'] % d['rollout_gpus_per_engine']:
        raise ValueError('GPU counts must be divisible by TP')
    preset = config['model']['args_file']
    if not isinstance(preset,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',preset) or not (ROOT / f'backends/slime/scripts/models/{preset}.sh').is_file():
        raise ValueError('unknown model preset')
    for key in ('tokenizer_digest', 'template_digest'):
        if not re.fullmatch(r'[0-9a-f]{64}', g[key]):
            raise ValueError(f'gateway.{key} must be a SHA-256 digest')
    workers = config['harbor']['workers']
    if (not isinstance(workers, list) or not workers or any(
            not isinstance(worker, str) or not re.fullmatch(r'[A-Za-z0-9_.:+@-]+', worker) or worker.startswith('-')
            for worker in workers)):
        raise ValueError('harbor.workers must contain nonempty SSH destinations')
    claude_config(read_json(config['harness']['profile']), g['advertised_url'], 'validation-only')
    return config


def catalog(config):
    from harborrl.trajectories.native import read_json
    from harborrl.data.harbor.native_inspector import inspect_native
    path = Path(config['tasks']['catalog'])
    entries = read_json(path)
    if not isinstance(entries,list) or not entries:
        raise ValueError('catalog must be a nonempty JSON list')
    seen = set()
    for entry in entries:
        if (set(entry) != {'id','revision','path','task_digest','reward_profile'}
                or not isinstance(entry['id'], str) or not entry['id'].strip()
                or not isinstance(entry['revision'], str) or not entry['revision'].strip()
                or entry['id'] in seen):
            raise ValueError('invalid or duplicate catalog identity')
        if (not isinstance(entry['path'], str) or not entry['path'].strip()
                or not isinstance(entry['task_digest'], str)
                or not re.fullmatch(r'[0-9a-f]{64}', entry['task_digest'])
                or not isinstance(entry['reward_profile'], dict)):
            raise ValueError(f'invalid native task lock: {entry["id"]}')
        seen.add(entry['id'])
        entry['path'] = str((path.parent / entry['path']).resolve())
        report = inspect_native(entry['path'], expected_digest=entry['task_digest'])
        if report['status'] != 'NEEDS_PROBE':
            raise ValueError(f"task preflight failed: {entry['id']}: {report['reasons']}")
        RewardProfile(**entry['reward_profile'])
    return entries


def launch_plan(config):
    from harborrl.config import ROOT
    from harborrl.backends.slime_v032.versions import runtime_contract_id
    catalog(config)
    c = config
    environment = {
        'DATASET': 'native', 'ALGO': 'grpo', 'HARNESS_OPTION': 'native-claude-code',
        'HF_CKPT': c['model']['checkpoint'], 'REF_LOAD': c['model']['reference'],
        'MODEL_ARGS_FILE': c['model']['args_file'], 'MODEL_TAG': c['model']['args_file'],
        'NUM_ROLLOUT': c['training']['num_rollout'], 'SAVE_INTERVAL': c['training']['save_interval'],
        'ROLLOUT_BATCH_SIZE': c['sampling']['groups_per_batch'],
        'N_SAMPLES': c['sampling']['group_size'],
        'ROLLOUT_MAX_RESPONSE_LEN': c['sampling']['max_tokens'],
        'ROLLOUT_MAX_CONTEXT_LEN': c['sampling']['max_context'],
        'NUM_GPUS': c['deployment']['num_gpus'], 'ACTOR_GPUS': c['deployment']['actor_gpus'],
        'ROLLOUT_GPUS': c['deployment']['rollout_gpus'],
        'TP_SIZE': c['deployment']['actor_tensor_parallel_size'],
        'ROLLOUT_NUM_GPUS_PER_ENGINE': c['deployment']['rollout_gpus_per_engine'],
        'HARBORRL_GPU_LAYOUT': c['deployment']['layout'],
        'HARBORRL_STRUCTURED_LAUNCH': '1', 'HARBORRL_VERIFY_POLICY_POOL': '1',
        'HARBORRL_SKIP_GLOBAL_CLEANUP': '1', 'SLIME_RAY_PLACEMENT_GPU_PROBE': '1',
        'SLIME_ENTRYPOINT': str(ROOT / 'backends/slime/train.py'),
        'PYTORCH_CUDA_ALLOC_CONF': ('max_split_size_mb:2048' if c['deployment']['layout'] == 'colocate'
                                    else 'max_split_size_mb:2048,expandable_segments:True'),
        'HARBORRL_NATIVE_ROLLOUT': '1',
        'HARBORRL_NATIVE_SLIME_CONTRACT': runtime_contract_id(c['training']['backend_contract']),
        'HARBORRL_NATIVE_RUNNER_PYTHON': c['harbor']['python'],
        'HARBORRL_NATIVE_RUNNER_WORKERS': json.dumps(c['harbor']['workers']),
        'HARBORRL_NATIVE_PROFILE': c['harness']['profile'],
        'HARBORRL_NATIVE_CATALOG': c['tasks']['catalog'],
        'HARBORRL_NATIVE_GATEWAY_HOST': c['gateway']['host'],
        'HARBORRL_NATIVE_GATEWAY_PORT': c['gateway']['port'],
        'HARBORRL_NATIVE_GATEWAY_URL': c['gateway']['advertised_url'],
        'HARBORRL_NATIVE_TOKENIZER_DIGEST': c['gateway']['tokenizer_digest'],
        'HARBORRL_NATIVE_TEMPLATE_DIGEST': c['gateway']['template_digest'],
        'HARBORRL_NATIVE_AUDITED_LOGPROBS': str(int(c['gateway']['audited_raw_logprobs'])),
        'HARBORRL_NATIVE_MAX_ATTEMPTS': c['sampling']['max_attempts'],
        'HARBORRL_NATIVE_DRAIN_TIMEOUT': '30',
        'RUNS_ROOT': c['output']['root'], 'WANDB_ENABLE': '0',
    }
    return {'config': config, 'command': [sys.executable, '-m', 'harborrl.platform.native_train'],
            'environment': {key: str(value) for key, value in environment.items()},
            'training_command': ['bash', str(ROOT / 'harborrl/platform/slime_train.sh')],
            'required_services': ['GPU Ray / SGLang / Messages Gateway', 'external CPU Harbor Docker workers'],
            'launcher': 'native Claude Code → Messages Gateway → Harbor verifier → Slime GRPO',
            'layout': {'mode': config['deployment']['layout'], 'offload': config['deployment']['layout']=='colocate'}}
