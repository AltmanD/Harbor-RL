"""Public entry point; optional runtime dependencies load only on execution."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

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
    checks.append({'dependency': 'nvidia-smi', 'ok': shutil.which('nvidia-smi') is not None})
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
        print(json.dumps({'checks': checks, 'scope': 'local dependencies and paths; GPU ABI and worker connectivity require runtime validation'}, indent=2))
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
    plan['code'] = {'commit': git('rev-parse', 'HEAD').decode().strip(), 'dirty': bool(git('status', '--porcelain')), 'patch': 'source.patch'}
    (run / 'launch.json').write_text(json.dumps(plan, indent=2) + '\n')
    # Only OS/runtime plumbing crosses this boundary. Training values come from the plan.
    allowed = {'PATH', 'HOME', 'USER', 'LANG', 'TMPDIR', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY'}
    env = {k:v for k,v in os.environ.items() if k in allowed}
    env.update(plan['environment'], RUN_DIR=str(run))
    return subprocess.call(plan['command'], cwd=ROOT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
