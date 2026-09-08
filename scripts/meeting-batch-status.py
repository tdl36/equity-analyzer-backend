#!/usr/bin/env python3
"""Read a conference batch without triggering downloads or paid generation."""
import argparse
import json
from pathlib import Path
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meeting_batch_readiness import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=lambda s: str(uuid.UUID(s)))
    parser.add_argument('--tickers', required=True, nargs='+', help='Entire original batch selection; prevents partial history from appearing complete')
    args = parser.parse_args()
    import requests
    from charlie_local_agent import CHARLIE_API, _agent_headers
    def get(path):
        response = requests.get(CHARLIE_API + path, headers=_agent_headers(), timeout=30)
        response.raise_for_status()
        return response.json()
    jobs = [j for j in get('/api/research/meeting-commands')['jobs'] if j.get('batch_id') == args.batch]
    if not jobs:
        raise ValueError('Batch not found in recent assignments; absence is not completion.')
    expected = {t.strip().upper() for t in args.tickers}
    if len(jobs) != len(expected) or {j['ticker'] for j in jobs} != expected:
        raise ValueError('Recent history does not contain the entire expected batch; cannot report readiness.')
    report = summarize([get('/api/research/assignments/' + j['id']) for j in jobs])
    report['batchId'] = args.batch
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'ready_for_review' else 2


if __name__ == '__main__':
    raise SystemExit(main())
