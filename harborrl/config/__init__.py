"""Explicit configuration for the interactive Slime launcher transition."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
from urllib.parse import urlsplit

import yaml

from harborrl.harnesses.identity import get_harness_descriptor

ROOT = Path(__file__).resolve().parents[2]
# Backend tuning remains explicit in the profile, using the launcher's names.
BACKEND_OPTIONS = frozenset("""
DATASET_DIR CUSTOM_CONFIG_PATH MAX_TURN ROLLOUT_MAX_RESPONSE_LEN
ROLLOUT_MAX_CONTEXT_LEN ROLLOUT_BATCH_SIZE N_SAMPLES NUM_ROLLOUT
ROLLOUT_TEMPERATURE MAX_TOKENS_PER_GPU OPTIMIZER_CPU_OFFLOAD USE_REMOTE_ENV
START_ENV_POOL_SERVER ENV_SERVER_URL CKPT_ROOT SAVE_CKPT RESUME_LOAD
MAX_CKPT_KEEP SAVE_INTERVAL CHECKPOINT_SAVE_FATAL CHECKPOINT_MIN_FREE_GB
CHECKPOINT_EXPECTED_GB HARBOR_VERSION_ENDPOINT SGLANG_RETURN_ORIGINAL_LOGPROB
EXTRA_GRPO_ARGS CLAUDE_CODE_CLI CLAUDE_CODE_LLM_BACKEND
CLAUDE_CODE_MARK_NON_TRAINABLE CLAUDE_CODE_MAX_TOOL_ROUNDS
CLAUDE_CODE_TURN_TIMEOUT_SEC CLAUDE_CODE_LOCAL_RUN_ROOT WANDB_ENABLE
WANDB_MODE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HOME TORCH_EXTENSIONS_DIR
TRITON_CACHE_DIR RAY_TMPDIR LD_LIBRARY_PATH SGLANG_MEM_FRACTION_STATIC SGLANG_DISABLE_CUDA_GRAPH SGLANG_ENABLE_WEIGHTS_CPU_BACKUP
""".split())
FIELDS = {
    'tasks': {'catalog'}, 'execution': {'backend'}, 'harness': {'name'},
    'model': {'provider', 'checkpoint', 'reference', 'args_file'},
    'training': {'backend', 'engine', 'algorithm', 'logprob_source', 'logprob_semantics'},
    'deployment': {'worker_urls', 'num_gpus', 'actor_gpus', 'rollout_gpus', 'layout', 'actor_tensor_parallel_size', 'rollout_gpus_per_engine'},
    'output': {'root'},
}


def load_config(path, overrides=()):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('configuration must be a mapping')
    if config.get("schema_version") == 2:
        from .native import FIELDS as native_fields, validate
        validate(config, path)
        for override in overrides:
            key, sep, value = override.partition("=")
            parts = key.split(".")
            if (not sep or len(parts) != 2 or parts[0] not in native_fields
                    or parts[1] not in native_fields[parts[0]] or not isinstance(config.get(parts[0]), dict)):
                raise ValueError(f"unknown native override: {key}")
            config[parts[0]][parts[1]] = yaml.safe_load(value)
        return validate(config, path)
    for override in overrides:
        key, sep, value = override.partition('=')
        parts = key.split('.')
        if not sep or len(parts) != 2 or parts[0] not in (set(FIELDS) | {'backend_options'}) or parts[1] not in (BACKEND_OPTIONS if parts[0] == 'backend_options' else FIELDS[parts[0]]):
            raise ValueError(f'unknown override: {key}')
        config.setdefault(parts[0], {})[parts[1]] = yaml.safe_load(value)
    if config.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    if set(config) - (set(FIELDS) | {'schema_version', 'backend_options'}):
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
    backend_options = config.get('backend_options', {})
    if not isinstance(backend_options, dict) or set(backend_options) - BACKEND_OPTIONS:
        raise ValueError('unknown backend_options; use documented Slime transition options')
    for key, value in backend_options.items():
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError(f'backend_options.{key} must be a string or number')
        backend_options[key] = str(expand(value))
    d = config.get('deployment')
    if isinstance(d, dict):
        d.setdefault('layout', 'split')
        d.setdefault('actor_tensor_parallel_size', d.get('actor_gpus'))
        d.setdefault('rollout_gpus_per_engine', d.get('rollout_gpus'))
    reserved = {'--colocate', '--offload', '--offload-train', '--offload-rollout',
                '--actor-num-nodes', '--actor-num-gpus-per-node', '--num-gpus-per-node',
                '--rollout-num-gpus', '--rollout-num-gpus-per-engine',
                '--tensor-model-parallel-size', '--pipeline-model-parallel-size',
                '--context-parallel-size', '--expert-model-parallel-size',
                '--expert-tensor-parallel-size', '--use-critic', '--prm-enable',
                '--critic-num-nodes', '--critic-num-gpus-per-node', '--prefill-num-servers',
                '--sglang-tensor-parallel-size', '--sglang-tp-size',
                '--sglang-data-parallel-size', '--sglang-dp-size',
                '--sglang-pipeline-parallel-size', '--sglang-pp-size',
                '--sglang-expert-parallel-size', '--sglang-ep-size',
                '--sglang-base-gpu-id', '--sglang-gpu-id-step'}
    for token in shlex.split(backend_options.get('EXTRA_GRPO_ARGS', '')):
        flag = token.split('=', 1)[0]
        if flag.startswith('--') and any(option.startswith(flag) for option in reserved):
            raise ValueError(f'layout parameter must be configured through deployment: {token}')
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
    for key in ('num_gpus', 'actor_gpus', 'rollout_gpus', 'actor_tensor_parallel_size', 'rollout_gpus_per_engine'):
        if type(d[key]) is not int or d[key] <= 0:
            raise ValueError(f'deployment.{key} must be a positive integer')
    if d['layout'] not in ('split', 'colocate'):
        raise ValueError('deployment.layout must be split or colocate')
    if d['layout'] == 'split' and d['actor_gpus'] + d['rollout_gpus'] > d['num_gpus']:
        raise ValueError('actor and rollout GPUs exceed the node budget')
    if d['layout'] == 'colocate' and not d['actor_gpus'] == d['rollout_gpus'] == d['num_gpus']:
        raise ValueError('colocate requires actor_gpus = rollout_gpus = num_gpus')
    if d['actor_gpus'] % d['actor_tensor_parallel_size'] or d['rollout_gpus'] % d['rollout_gpus_per_engine']:
        raise ValueError('GPU counts must be divisible by their tensor parallel sizes')
    if d['rollout_gpus'] > d['rollout_gpus_per_engine'] and backend_options.get('HARBOR_VERSION_ENDPOINT'):
        raise ValueError('multiple engines require pool version validation; remove HARBOR_VERSION_ENDPOINT')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', str(config['model']['args_file'])):
        raise ValueError('model.args_file must name a bundled model preset')
    return config


def launch_plan(config):
    if config.get("schema_version") == 2:
        from .native import launch_plan as native_plan
        return native_plan(config)
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
        'TP_SIZE': c['deployment']['actor_tensor_parallel_size'],
        'HARBORRL_GPU_LAYOUT': c['deployment']['layout'],
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:2048' if c['deployment']['layout'] == 'colocate' else 'max_split_size_mb:2048,expandable_segments:True',
        'HARBORRL_STRUCTURED_LAUNCH': '1',
        'SLIME_RAY_PLACEMENT_GPU_PROBE': '1',
        'HARBORRL_VERIFY_POLICY_POOL': '1',
        'ROLLOUT_NUM_GPUS_PER_ENGINE': c['deployment']['rollout_gpus_per_engine'],
        'RUNS_ROOT': c['output']['root'], 'HARBOR_IR_ROOT': str(Path(c['output']['root']) / 'trajectories'),
        'HARBOR_LOGPROB_SOURCE': c['training']['logprob_source'],
        'HARBOR_LOGPROB_SEMANTICS': 'raw_model',
        'SLIME_ENTRYPOINT': str(ROOT / 'backends/slime/train.py'),
        'EXPLORE_INTRINSIC': '0', 'EXPLORE_AGENT57_LITE': '0', 'DAPO_OVERLONG_BUFFER_ENABLE': '0',
        'HARBORRL_SKIP_GLOBAL_CLEANUP': '1',
    }
    env.update(c.get('backend_options', {}))
    return {'config': c, 'layout': {
                'mode': c['deployment']['layout'],
                'reserved_gpus': c['deployment']['actor_gpus'] + (c['deployment']['rollout_gpus'] if c['deployment']['layout'] == 'split' else 0),
                'actor_dp': c['deployment']['actor_gpus'] // c['deployment']['actor_tensor_parallel_size'],
                'rollout_engines': c['deployment']['rollout_gpus'] // c['deployment']['rollout_gpus_per_engine'],
                'offload': c['deployment']['layout'] == 'colocate'}, 'environment': {k: str(v) for k,v in env.items()},
            'command': ['bash', str(ROOT / 'harborrl/platform/slime_train.sh')],
            'required_services': ['environment worker', 'local Ray and SGLang (launcher managed)'],
            'launcher': 'legacy Slime shell; unmigrated tuning retains backend defaults'}
