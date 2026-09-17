import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from harborrl.config import load_config, launch_plan
from harborrl.cli import main


def profile(tmp_path):
    data = yaml.safe_load(Path('configs/train_interactive_smoke.yaml').read_text())
    data['tasks']['catalog'] = './catalog.jsonl'
    data['model'].update(checkpoint='./model', reference='./reference')
    data['training']['logprob_source'] = 'audited-test-version'
    data['deployment']['worker_urls'] = 'http://localhost:18081'
    p = tmp_path / 'config.yaml'
    p.write_text(yaml.safe_dump(data))
    return p


def test_config_override_and_no_ambient_training_override(tmp_path, monkeypatch):
    p = profile(tmp_path)
    monkeypatch.setenv('HF_CKPT', '/wrong')
    c = load_config(p, ['harness.name=claude-code'])
    plan = launch_plan(c)
    assert plan['environment']['HARNESS_OPTION'] == 'claude-code'
    assert plan['environment']['HF_CKPT'] == str(tmp_path / 'model')
    with pytest.raises(ValueError, match='unsupported'):
        load_config(p, ['execution.backend=harbor_native'])
    with pytest.raises(ValueError, match='unknown override'):
        load_config(p, ['model.typo=x'])


def test_dry_run_has_no_launch_or_output(tmp_path, monkeypatch, capsys):
    p = profile(tmp_path)
    monkeypatch.setattr(subprocess, 'call', lambda *a, **k: pytest.fail('launched process'))
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('launched process'))
    before = sorted(tmp_path.rglob('*'))
    assert main(['train', '--config', str(p), '--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['config']['execution']['backend'] == 'lightrl_interactive'
    assert before == sorted(tmp_path.rglob('*'))


def test_task_helpers_import_without_training_dependencies():
    script = '''
import sys
before = set(sys.modules)
from harborrl.tasks.metadata import _make_task_spec
from harborrl.tasks.cli import main
assert _make_task_spec({'task_name': 'x'}).task_name == 'x'
assert not any(m in set(sys.modules) - before for m in ['torch', 'ray', 'slime', 'camel'])
'''
    subprocess.run([sys.executable, '-c', script], check=True)


def test_harness_constructor_rejects_unknown_and_missing(monkeypatch):
    from types import SimpleNamespace
    from harborrl.harnesses import factory
    class Harness:
        def __init__(self, *, model_type):
            self.model_type = model_type
    monkeypatch.setattr(factory, 'import_module', lambda _: SimpleNamespace(CamelAgent=Harness))
    assert factory.create_harness('camel', model_type='x', tool_schemas=[]).model_type == 'x'
    with pytest.raises(ValueError, match='missing'):
        factory.create_harness('camel')
    with pytest.raises(ValueError, match='unexpected'):
        factory.create_harness('camel', model_type='x', typo=True)


def test_static_inspection_preserves_attempts_without_readiness(tmp_path, capsys):
    from harborrl.tasks.cli import main as hub
    task = tmp_path / 'tasks' / 'example'
    (task / 'environment').mkdir(parents=True)
    (task / 'tests').mkdir()
    (task / 'task.toml').write_text('[environment]\n')
    (task / 'instruction.md').write_text('Create a file.')
    (task / 'environment/Dockerfile').write_text('FROM alpine\n')
    (task / 'tests/test.sh').write_text('echo 0 > /logs/verifier/reward.txt\n')
    run = tmp_path / 'report'
    args = ['inspect', '--tasks-dir', str(task.parent), '--dataset', 'test',
            '--revision', 'abc', '--run-dir', str(run)]
    assert hub(args) == 0
    assert hub(args) == 0
    attempts = list(run.glob('*.jsonl'))
    assert len(attempts) == 2
    row = json.loads(attempts[0].read_text())
    assert row['status'] == 'NEEDS_PROBE'
    assert row['trainable'] is False and row['evaluable'] is False
    assert row['ref']['source_digest']


def test_missing_interpolation_is_explicit(tmp_path, monkeypatch):
    p = profile(tmp_path)
    data = yaml.safe_load(p.read_text())
    data['model']['checkpoint'] = '${MISSING_TEST_MODEL}'
    p.write_text(yaml.safe_dump(data))
    monkeypatch.delenv('MISSING_TEST_MODEL', raising=False)
    with pytest.raises(ValueError, match='missing declared environment'):
        load_config(p)


def test_explicit_backend_options_do_not_accept_model_overrides(tmp_path):
    p = profile(tmp_path)
    config = load_config(p, ['backend_options.N_SAMPLES=8'])
    assert launch_plan(config)['environment']['N_SAMPLES'] == '8'
    with pytest.raises(ValueError, match='unknown override'):
        load_config(p, ['backend_options.HF_CKPT=/wrong'])


def test_ray_status_accepts_current_cli_success():
    source = Path('harborrl/platform/slime_train/lib_launch.sh').read_text()
    start = source.index('  RAY_STATUS_LOWER=$(echo')
    end = source.index('  if (( status_attempt', start)
    block = source[start:end]
    script = "RAY_JOB_SUBMISSION_ID=test-job\nRAY_STATUS_OUTPUT=\"Job 'test-job' succeeded\"\nfor i in 1; do\n" + block + "\ndone\n[[ $RAY_STATUS_STATE == succeeded ]]"
    subprocess.run(['bash', '-c', script], check=True)
