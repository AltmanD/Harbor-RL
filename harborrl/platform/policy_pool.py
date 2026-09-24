"""Small, dependency-free contract for a synchronized inference pool."""
import json
import os
import time
from pathlib import Path


def validate_pool(states, expected=None):
    if not states or any(not s.get('healthy') or s.get('weight_version') in (None, '') for s in states):
        raise RuntimeError('policy pool contains an unhealthy or unversioned engine')
    endpoints = [s.get('endpoint') for s in states]
    if any(endpoints) and len(set(endpoints)) != len(endpoints):
        raise RuntimeError('policy pool contains duplicate endpoints')
    versions = {str(s['weight_version']) for s in states}
    if len(versions) != 1 or (expected is not None and versions != {str(expected)}):
        raise RuntimeError(f'policy pool version mismatch: {versions}, expected={expected}')
    return versions.pop()


def publish_pool(states, expected=None):
    version = validate_pool(states, expected)
    root = Path(os.environ['RUN_DIR'])
    payload = {'weight_version': version, 'engines': states, 'timestamp': time.time()}
    temp = root / 'policy-pool.tmp'
    temp.write_text(json.dumps(payload) + '\n')
    temp.replace(root / 'policy-pool.json')
    with (root / 'policy-pool-history.jsonl').open('a') as f:
        f.write(json.dumps(payload) + '\n')
    return version
