"""Record and close only environment leases acquired by the current run."""
import hashlib
import json
import os
from pathlib import Path
from urllib import request, error
from uuid import uuid4


def record_lease(endpoint, lease_id):
    if os.getenv('HARBORRL_VERIFY_POLICY_POOL') != '1':
        return
    root = Path(os.environ['RUN_DIR']) / 'leases'
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f'{endpoint}:{lease_id}'.encode()).hexdigest()
    temp = root / f'{key}.{uuid4().hex}.tmp'
    try:
        temp.write_text(json.dumps({'endpoint': endpoint, 'lease_id': lease_id}))
        temp.replace(root / f'{key}.json')
    finally:
        temp.unlink(missing_ok=True)


def close_run_leases(root):
    opener = request.build_opener(request.ProxyHandler({}))
    outcomes = []
    for path in sorted((Path(root) / 'leases').glob('*.json')):
        try:
            lease = json.loads(path.read_text())
            if not all(isinstance(lease[key], str) and lease[key] for key in ('endpoint', 'lease_id')):
                raise ValueError('lease endpoint and ID must be nonempty strings')
            req = request.Request(lease['endpoint'].rstrip('/') + '/close',
                                  data=json.dumps({'lease_id': lease['lease_id']}).encode(),
                                  headers={'Content-Type': 'application/json'})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            outcomes.append({'record': path.name, 'error': str(exc)})
            (Path(root) / 'lease-cleanup.json').write_text(json.dumps(outcomes, indent=2) + '\n')
            continue
        try:
            with opener.open(req, timeout=15) as response:
                result = json.load(response)
            outcomes.append({'lease_id': lease['lease_id'], 'result': result})
        except error.HTTPError as exc:
            outcomes.append({'lease_id': lease['lease_id'], 'already_closed': True} if exc.code == 410
                            else {'lease_id': lease['lease_id'], 'error': str(exc)})
        except Exception as exc:
            outcomes.append({'lease_id': lease['lease_id'], 'error': str(exc)})
        (Path(root) / 'lease-cleanup.json').write_text(json.dumps(outcomes, indent=2) + '\n')
    (Path(root) / 'lease-cleanup.json').write_text(json.dumps(outcomes, indent=2) + '\n')


if __name__ == '__main__':
    close_run_leases(os.environ['RUN_DIR'])
