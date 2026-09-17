"""Public entry point; optional runtime dependencies load only on execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid
from urllib import request

from harborrl.config import ROOT, launch_plan, load_config


def doctor(plan):
    checks = []
    for package in ('ray', 'torch', 'sglang', 'transformers'):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        checks.append({'dependency': package, 'version': version, 'ok': version is not None})
    checks.append({'dependency': 'bundled Megatron-LM', 'ok': (ROOT / 'backends/Megatron-LM/megatron/core/__init__.py').is_file()})
    for key in ('HF_CKPT', 'REF_LOAD', 'ROLLOUT_PROMPT_DATA'):
        checks.append({'path': key, 'ok': Path(plan['environment'][key]).exists()})
    model_config = Path(plan['environment']['HF_CKPT']) / 'config.json'
    if model_config.is_file():
        model = json.loads(model_config.read_text())
        for tp in {int(plan['environment']['TP_SIZE']), int(plan['environment']['ROLLOUT_NUM_GPUS_PER_ENGINE'])}:
            heads = model.get('num_attention_heads', 0)
            kv_heads = model.get('num_key_value_heads', heads)
            checks.append({'model_tp': tp, 'ok': bool(heads and kv_heads and heads % tp == 0 and (kv_heads % tp == 0 or tp % kv_heads == 0))})
    if plan['layout']['offload']:
        spec = importlib.util.find_spec('torch_memory_saver')
        checks.append({'dependency': 'torch_memory_saver', 'ok': spec is not None})
        if spec and spec.origin:
            preload = Path(spec.origin).parent.parent / 'torch_memory_saver_hook_mode_preload.abi3.so'
            probe_env = dict(os.environ)
            probe_env.update(plan['environment'], LD_PRELOAD=str(preload))
            try:
                probe = subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__)'],
                                       env=probe_env, capture_output=True, text=True, timeout=60)
                checks.append({'dependency': 'memory saver CUDA preload ABI', 'ok': probe.returncode == 0,
                               'detail': (probe.stdout if probe.returncode == 0 else probe.stderr)[-1500:]})
            except subprocess.TimeoutExpired:
                checks.append({'dependency': 'memory saver CUDA preload ABI', 'ok': False, 'detail': 'probe timed out'})
    checks.append({'dependency': 'nvidia-smi', 'ok': shutil.which('nvidia-smi') is not None})
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10, check=True)
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            count = len([device for device in visible.split(',') if device.strip() and device.strip() != '-1']) if visible is not None else len(result.stdout.splitlines())
            checks.append({'resource': 'visible GPU budget', 'available': count,
                           'ok': count >= plan['config']['deployment']['num_gpus']})
        except (subprocess.SubprocessError, OSError) as exc:
            checks.append({'resource': 'visible GPU budget', 'ok': False, 'detail': str(exc)})
    opener = request.build_opener(request.ProxyHandler({}))
    for worker in plan['config']['deployment']['worker_urls'].split(','):
        try:
            with opener.open(worker.strip().rstrip('/') + '/readyz', timeout=10) as response:
                ready = json.load(response)
            checks.append({'service': worker, 'ok': ready.get('ok') is True})
        except (OSError, ValueError) as exc:
            checks.append({'service': worker, 'ok': False, 'detail': str(exc)})

    if plan['config']['harness']['name'] == 'claude_code_cli':
        checks.append({'dependency': 'claude CLI', 'ok': shutil.which('claude') is not None})
    else:
        checks.append({'dependency': 'camel-ai', 'ok': importlib.util.find_spec('camel') is not None})
    return checks


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'eval':
        return subprocess.call([sys.executable, '-m', 'tools.evaluation', 'run', *argv[1:]])
    if argv and argv[0] == 'hub':
        from harborrl.tasks.cli import main as hub_main
        return hub_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'doctor'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = launch_plan(load_config(args.config, args.set))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    checks = doctor(plan)
    if args.command == 'doctor':
        print(json.dumps({'checks': checks, 'scope': 'dependencies, preload ABI, GPU budget and worker readiness; full memory capacity and model execution require runtime validation'}, indent=2))
        return int(any(not c['ok'] for c in checks))
    if any(not c['ok'] for c in checks):
        print(json.dumps({'failed_checks': [c for c in checks if not c['ok']]}, indent=2), file=sys.stderr)
        return 1
    run = Path(plan['config']['output']['root']) / 'training' / uuid.uuid4().hex
    run.mkdir(parents=True)
    plan['environment'].update(RUN_DIR=str(run), RUN_ID=run.name, RUN_NAME=run.name)
    plan['environment']['HARBOR_IR_ROOT'] = str(run / 'ir')
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT)
    patch = git('diff', 'HEAD', '--', 'harborrl', 'backends', 'pyproject.toml')
    (run / 'source.patch').write_bytes(patch)
    source_hashes = {}
    source_files = git('ls-files', '-z', '--cached', '--others', '--exclude-standard', '--',
                       'harborrl', 'backends', 'pyproject.toml').decode().split('\0')
    untracked = set(git('ls-files', '-z', '--others', '--exclude-standard', '--',
                        'harborrl', 'backends').decode().split('\0'))
    for name in source_files:
        source = ROOT / name
        if not name or not source.is_file():
            continue
        content = source.read_bytes()
        source_hashes[name] = hashlib.sha256(content).hexdigest()
        if name in untracked:
            target = run / 'source-new' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    (run / 'source-hashes.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
    plan['code'] = {'commit': git('rev-parse', 'HEAD').decode().strip(), 'dirty': bool(git('status', '--porcelain')), 'patch': 'source.patch'}
    (run / 'launch.json').write_text(json.dumps(plan, indent=2) + '\n')
    # Only OS/runtime plumbing crosses this boundary. Training values come from the plan.
    allowed = {'PATH', 'HOME', 'USER', 'LANG', 'TMPDIR', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY'}
    env = {k:v for k,v in os.environ.items() if k in allowed}
    env.update(plan['environment'], RUN_DIR=str(run))
    child = subprocess.Popen(plan['command'], cwd=ROOT, env=env, start_new_session=True)
    def forward_signal(signum, frame):
        if child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass
    previous = {sig: signal.signal(sig, forward_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return child.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
