"""Durable local recap outbox. A backend outage must not discard paid output."""
import json
import os
from pathlib import Path
import uuid
import requests

OUTBOX = Path.home() / 'Library/Application Support/charlie-agent/recap-outbox'


def save_result(job_id, result, root=OUTBOX):
    name = str(uuid.UUID(job_id))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    target = root / (name + '.json')
    temp = root / ('.' + name + '.tmp')
    with temp.open('w') as stream:
        os.chmod(temp, 0o600)
        json.dump({'jobId': name, 'result': result}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target)
    return target


def deliver(path, api, headers, post=None):
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
        job_id = str(uuid.UUID(payload['jobId']))
        response = (post or requests.post)(api + '/api/pipeline/jobs/' + job_id + '/result',
            headers=headers, json={'status': 'complete', 'result': payload['result']}, timeout=30)
        if not response.ok:
            return False
        path.unlink(missing_ok=True)
        return True
    except (OSError, ValueError, requests.RequestException):
        return False


def flush(api, headers, root=OUTBOX):
    for path in sorted(Path(root).glob('*.json'))[:3]:
        deliver(path, api, headers)
