import json

import pytest

from harborrl.config import launch_plan, load_config
from harborrl.platform.policy_pool import validate_pool
from test_cleanup_entrypoints import profile


@pytest.mark.parametrize('layout,actor,rollout,tp,engine,dp,count,reserved', [
    ('split',4,4,4,4,1,1,8), ('split',6,2,2,2,3,1,8),
    ('split',2,6,2,2,1,3,8), ('split',4,2,4,2,1,1,6),
    ('colocate',8,8,4,4,2,2,8),
])
def test_layout_plan(tmp_path,layout,actor,rollout,tp,engine,dp,count,reserved):
    cfg=load_config(profile(tmp_path), [f'deployment.{k}={v}' for k,v in {
        'layout':layout,'num_gpus':8,'actor_gpus':actor,'rollout_gpus':rollout,
        'actor_tensor_parallel_size':tp,'rollout_gpus_per_engine':engine}.items()])
    plan=launch_plan(cfg)
    assert plan['layout']['actor_dp']==dp
    assert plan['layout']['rollout_engines']==count
    assert plan['layout']['reserved_gpus']==reserved
    assert plan['environment']['TP_SIZE']==str(tp)


@pytest.mark.parametrize('overrides', [
    ['deployment.layout=colocate'],
    ['deployment.actor_gpus=6','deployment.num_gpus=8','deployment.rollout_gpus=2','deployment.actor_tensor_parallel_size=4'],
    ['backend_options.EXTRA_GRPO_ARGS=--colocate'],
    ['backend_options.EXTRA_GRPO_ARGS=--tensor-model-parallel-size=8'],
    ['backend_options.EXTRA_GRPO_ARGS=--tensor-model-parallel=8'],
    ['backend_options.EXTRA_GRPO_ARGS=--sglang-pp-size=2'],
    ['deployment.rollout_gpus_per_engine=1','backend_options.HARBOR_VERSION_ENDPOINT=http://one-engine'],
])
def test_invalid_layout(tmp_path,overrides):
    with pytest.raises(ValueError):load_config(profile(tmp_path),overrides)


def test_policy_pool_rejects_partial_updates():
    states=[{'healthy':True,'weight_version':'2'},{'healthy':True,'weight_version':'2'}]
    assert validate_pool(states,'2')=='2'
    with pytest.raises(RuntimeError):validate_pool(states,'3')
    states[1]['weight_version']='1'
    with pytest.raises(RuntimeError):validate_pool(states)
    with pytest.raises(RuntimeError):validate_pool([])
    with pytest.raises(RuntimeError):validate_pool([{'healthy':False,'weight_version':'2'}])


def test_colocate_allocator_and_real_mapping_probe(tmp_path):
    c=load_config(profile(tmp_path), ['deployment.layout=colocate','deployment.actor_gpus=4','deployment.rollout_gpus=4'])
    env=launch_plan(c)['environment']
    assert 'expandable_segments:True' not in env['PYTORCH_CUDA_ALLOC_CONF']
    assert env['SLIME_RAY_PLACEMENT_GPU_PROBE']=='1'


def test_cleanup_closes_only_recorded_leases(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from harborrl.platform.run_leases import record_lease, close_run_leases

    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('RUN_DIR', str(tmp_path))
    monkeypatch.setenv('HARBORRL_VERIFY_POLICY_POOL', '1')
    try:
        record_lease(f'http://127.0.0.1:{server.server_port}', 'owned-lease')
        close_run_leases(tmp_path)
        assert received == [{'lease_id': 'owned-lease'}]
        assert json.loads((tmp_path / 'lease-cleanup.json').read_text())[0]['result']['ok']
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_phase_failure_is_recorded_and_propagated(tmp_path, monkeypatch):
    import ast
    import os
    import time
    from pathlib import Path
    from contextlib import contextmanager

    # Exercise the driver boundary without importing CUDA/Ray on the CPU host.
    tree = ast.parse(Path('backends/slime/train.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'layout_phase')
    namespace = dict(os=os, time=time, Path=Path, json=json, contextmanager=contextmanager)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'train.py', 'exec'), namespace)
    monkeypatch.setenv('RUN_DIR', str(tmp_path))
    monkeypatch.setenv('HARBORRL_VERIFY_POLICY_POOL', '1')
    advanced = False
    with pytest.raises(TimeoutError, match='restore failed'):
        with namespace['layout_phase']('rollout_kv_onload'):
            raise TimeoutError('restore failed')
        advanced = True
    assert not advanced
    row = json.loads((tmp_path / 'layout-phases.jsonl').read_text())
    assert row['status'] == 'failed'
    assert row['phase'] == 'rollout_kv_onload'


def test_launcher_cancel_stops_only_its_ray_job(tmp_path):
    import os
    import signal
    import subprocess
    import time
    from pathlib import Path

    library = Path('harborrl/platform/slime_train/lib_launch.sh').read_text()
    prologue = library.split('ROUTER_LOG=', 1)[0]
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_ray = fake_bin / 'ray'
    fake_ray.write_text('#!/bin/sh\nprintf "%s\\n" "$*" > "$RAY_CALL_LOG"\n')
    fake_ray.chmod(0o755)
    ready = tmp_path / 'ready'
    script = tmp_path / 'cancel.sh'
    script.write_text('set -eu\n' + prologue + '\ntouch "$READY_FILE"\nwhile true; do sleep 0.1; done\n')
    env = dict(os.environ, PATH=str(fake_bin) + ':' + os.environ['PATH'],
               RAY_CALL_LOG=str(tmp_path / 'ray-call'), READY_FILE=str(ready),
               MASTER_ADDR='127.0.0.1', RAY_JOB_SUBMISSION_ID='owned-job',
               HARBORRL_STRUCTURED_LAUNCH='0')
    process = subprocess.Popen(['bash', str(script)], env=env)
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=3) == 143
        assert (tmp_path / 'ray-call').read_text().strip() == 'job stop --address=http://127.0.0.1:8265 owned-job'
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_reward_filtered_batch_padding_preserves_real_signal():
    from harborrl.platform.dp_batch import pad_filtered_batch
    from copy import deepcopy
    data = {'tokens': [[1, 2]] * 17, 'response_lengths': [1] * 17,
            'loss_masks': [[1] for _ in range(17)], 'rewards': list(range(17)),
            'per_is_weights': [1.] * 17, 'sample_indices': list(range(17))}
    original = deepcopy(data)
    assert pad_filtered_batch(data, 16) == 15
    for key in original:
        assert data[key][:17] == original[key]
        assert len(data[key]) == 32
    assert sum(sum(m) for m in data['loss_masks']) == 17
    assert data['rewards'][17:] == [0.] * 15
    assert pad_filtered_batch(data, 16) == 0
