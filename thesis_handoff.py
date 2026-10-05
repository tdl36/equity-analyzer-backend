"""Private, resumable assistant-to-Charlie draft handoff. Never approves or calls models."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid
import urllib.error
import urllib.request
from thesis_imports import normalize, digest

API = 'https://equity-analyzer-backend.onrender.com/api/thesis-imports'
APP = 'https://charlie-deployment.tonydlee.workers.dev/'
EXTENSIONS = {'.pdf', '.docx', '.txt', '.md', '.csv', '.xlsx', '.zip'}


def save(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.handoff-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read(path):
    return json.loads(Path(path).read_text())


def secret():
    key = os.environ.get('CHARLIE_API_KEY', '').strip()
    if not key:
        try:
            result = subprocess.run(['security', 'find-generic-password', '-s', 'charlie-agent',
                                     '-a', 'CHARLIE_API_KEY', '-w'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                key = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    config = Path.home() / '.charlie_agent_config.json'
    if not key and config.exists():
        data = read(config)
        key = data.get('CHARLIE_API_KEY') or data.get('charlie_api_key', '')
    if not key:
        raise ValueError('Charlie access is unavailable. Configure CHARLIE_API_KEY in the existing Mac agent credentials.')
    return key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Charlie redirected the request; credentials were not forwarded.')


def request(path='', body=None):
    # Fixed origin and bounded routes prevent forwarding the private key elsewhere.
    if not re.fullmatch(r'(?:/prepare/[A-Z0-9][A-Z0-9.-]{0,19}|/[a-f0-9-]{36})?', path):
        raise ValueError('Unsupported draft handoff route.')
    data = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
    req = urllib.request.Request(API + path, data=data, headers={
        'Authorization': 'ApiKey ' + secret(), 'Content-Type': 'application/json'})
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=45) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'Charlie returned HTTP {exc.code}. The saved submission can be retried; check access or reconcile a changed baseline.') from None
    except urllib.error.URLError:
        raise ValueError('Charlie is unreachable. Retry the same handoff; its saved submission is preserved.') from None


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(ticker, folder, output, call=request):
    ticker = ticker.upper().strip()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,19}', ticker):
        raise ValueError('Invalid ticker.')
    folder = Path(folder).expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError('Choose a source folder.')
    output = Path(output).expanduser().resolve()
    if output == folder or folder in output.parents:
        raise ValueError('Keep the private handoff outside the source folder.')
    preparation = call('/prepare/' + ticker)
    baseline = preparation.get('package', {}).get('baseline', {})
    if not re.fullmatch('[a-f0-9]{64}', baseline.get('hash', '')) or baseline.get('mode') not in ('initial', 'upgrade'):
        raise ValueError('Charlie did not return a valid preparation baseline.')
    manifest = []
    for path in sorted(folder.rglob('*')):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in EXTENSIONS:
            resolved = path.resolve()
            if folder not in resolved.parents:
                continue
            manifest.append({'path': str(resolved), 'sha256': file_hash(resolved),
                             'reviewStatus': 'not attested; inventory only'})
    if not manifest:
        raise ValueError('No supported source files found.')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    save(output / 'preparation.json', preparation)
    save(output / 'state.json', {'ticker': ticker, 'sourceFolder': str(folder), 'files': manifest})
    return {'workspace': str(output), 'sourceFiles': len(manifest), 'mode': baseline['mode'],
            'next': 'Read the sources, retain preparation.package.baseline and stable IDs, and write draft.json. Submit only after completing the research.'}


def submit(workspace, draft=None, call=request):
    workspace = Path(workspace).expanduser().resolve(strict=True)
    with (workspace / '.submit.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('This handoff is already submitting. Wait for that attempt, then retry.') from None
        return _submit(workspace, draft, call)


def _submit(workspace, draft=None, call=request):
    workspace = Path(workspace).expanduser().resolve(strict=True)
    state = read(workspace / 'state.json')
    frozen = workspace / 'submission.json'
    if frozen.exists():
        package = read(frozen)
        if draft is not None and digest(normalize(read(draft))) != digest(package):
            raise ValueError('A different submission is already frozen. Use a new preparation workspace for revisions.')
    else:
        package = normalize(read(draft or workspace / 'draft.json'))
        baseline = read(workspace / 'preparation.json')['package']['baseline']
        if package['ticker'] != state['ticker'] or package['baseline'] != baseline:
            raise ValueError('Ticker or baseline differs from preparation. Reconcile the draft with the frozen preparation before submitting.')
        for item in state['files']:
            path = Path(item['path'])
            if not path.is_file() or file_hash(path) != item['sha256']:
                raise ValueError('A source file changed or is unavailable. Reconcile sources in a new preparation workspace.')
        save(frozen, package)  # Persist exact retry bytes before any remote mutation.
    package = normalize(package)
    if package['ticker'] != state['ticker']:
        raise ValueError('Saved submission ticker does not match the workspace.')
    result = call('', package)
    identifier = str(uuid.UUID(result['id']))
    detail = call('/' + identifier)
    if detail['ticker'] != package['ticker'] or detail['fingerprint'] != digest(package):
        raise ValueError('Charlie receipt does not match this draft. Submission preserved for investigation.')
    receipt = {'id': identifier, 'ticker': package['ticker'], 'status': detail['status'],
               'fingerprint': detail['fingerprint'], 'stale': detail['stale'],
               'reviewUrl': APP + '?thesisDraft=' + identifier + '#view=thesisimports&ticker=' + package['ticker']}
    save(workspace / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('ticker'); p.add_argument('folder'); p.add_argument('--output', required=True)
    s = commands.add_parser('submit')
    s.add_argument('workspace'); s.add_argument('--draft')
    args = parser.parse_args()
    try:
        result = prepare(args.ticker, args.folder, args.output) if args.command == 'prepare' else submit(args.workspace, args.draft)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        # Paths/credentials/source bodies from low-level exceptions must not leak.
        parser.exit(1, (str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError)
                        else 'Handoff could not read or save its files. Check the private workspace and retry.') + '\n')


if __name__ == '__main__':
    main()
