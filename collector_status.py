"""Loopback collection monitor and authenticated same-origin refresh controls."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sqlite3
import secrets
import threading
import time
from collection_refresh import RefreshManager
from charlie_collector import Collector
from urllib.parse import urlsplit
from charlie_collector import DEFAULT_STATE, now, handoff_fingerprint


def destination_label(doc):
    if doc['usage'] == 'reference_only':
        return 'Private staging · excluded from AI ingestion'
    if not doc['destination']:
        return 'Awaiting iCloud handoff'
    parts = Path(doc['destination']).parts
    for root in ('STOCKS', 'CATALYSTS'):
        if root in parts:
            index = parts.index(root)
            if len(parts) > index + 2 and parts[index + 1] == doc['ticker']:
                return '/'.join(parts[index:-1]) + '/'
    return 'Existing library location'


def snapshot(state):
    path = Path(state) / 'ledger.sqlite3'
    if not path.exists():
        return {'checked': now(), 'runs': []}
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        runs = []
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for run in db.execute('SELECT * FROM runs ORDER BY created DESC LIMIT 20'):
            item = dict(run)
            for table in ('tasks', 'documents', 'observations'):
                item[table] = [dict(r) for r in db.execute(f'SELECT * FROM {table} WHERE run=?', (run['id'],))]
            item['verifications'] = []
            if 'verifications' in tables:
                for row in db.execute('SELECT * FROM verifications WHERE run=? ORDER BY ticker', (run['id'],)):
                    v = dict(row)
                    docs = [d for d in item['documents'] if d['ticker'] == v['ticker']]
                    v['current'] = v.pop('fingerprint') == handoff_fingerprint(docs)
                    v['missing'] = len(json.loads(v['missing']))
                    item['verifications'].append(v)
            for doc in item['documents']:
                doc['destinationFolder'] = destination_label(doc)
                doc.pop('staged', None)
                doc['destination'] = Path(doc['destination']).name if doc['destination'] else None
            runs.append(item)
        all_tasks = [dict(r) for r in db.execute('SELECT ticker,kind,status FROM tasks')]
        tickers = sorted({t['ticker'] for t in all_tasks})
        completed = [t for t in tickers if all(any(x['ticker']==t and x['kind']==kind and x['status'] in ('complete','complete_with_exceptions','no_results') for x in all_tasks) for kind in ('transcript','broker-report'))]
        return {'checked': now(), 'runs': runs, 'coverage': {'tracked': len(tickers), 'everReviewed': len(completed), 'remaining': [t for t in tickers if t not in completed]}}
    finally:
        db.close()


def handler(state, port):
    csrf = secrets.token_urlsafe(32)
    class Monitor(BaseHTTPRequestHandler):
        def do_POST(self):
            expected = (f'http://127.0.0.1:{port}', f'http://localhost:{port}')
            if self.headers.get('Host') not in (f'127.0.0.1:{port}',f'localhost:{port}') or self.headers.get('Origin') not in expected or not secrets.compare_digest(self.headers.get('X-Refresh-Token',''), csrf):
                self.send_error(403); return
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 16000 or self.headers.get('Content-Type','').split(';')[0] != 'application/json':
                    self.send_error(400); return
                data = json.loads(self.rfile.read(length))
            except (ValueError,TypeError):
                self.send_error(400); return
            c = None
            try:
                if not isinstance(data,dict): raise ValueError('Expected an object')
                c = Collector(state); manager = RefreshManager(c)
                path = urlsplit(self.path).path
                if path == '/api/refresh/policy': result = manager.save(data)
                elif path == '/api/refresh/trigger': result = {'requestId':manager.trigger(data.get('ticker'))}
                elif path == '/api/refresh/retry': result = manager.retry(data.get('requestId'))
                elif path == '/api/refresh/cancel': result = manager.cancel(data.get('requestId'))
                else: self.send_error(404); return
                body = json.dumps({'ok':True,'result':result}).encode()
                code = 200
            except (ValueError,TypeError,KeyError,OSError,sqlite3.Error) as exc:
                body = json.dumps({'error':str(exc)[:500]}).encode();code = 400
            finally:
                if c: c.db.close()
            self.send_response(code);self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            # Reject rebinding/foreign-origin access. No CORS; write routes additionally require the per-process token.
            if self.headers.get('Host') not in (f'127.0.0.1:{port}', f'localhost:{port}'):
                self.send_error(403)
                return
            origin = self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            if path == '/api/status':
                try:
                    result = snapshot(state)
                    c = Collector(state)
                    try: result['refresh'] = RefreshManager(c).status()
                    finally: c.db.close()
                    result['refreshToken'] = csrf
                    data = json.dumps(result).encode()
                except sqlite3.Error:
                    self.send_error(503, 'Collection ledger temporarily unavailable')
                    return
                mime = 'application/json'
            elif path == '/':
                data = (Path(__file__).parent / 'src/collector-monitor.html').read_bytes()
                mime = 'text/html; charset=utf-8'
            elif path == '/collection-controls.js':
                data = (Path(__file__).parent / 'src/collection-controls.js').read_bytes()
                mime = 'text/javascript; charset=utf-8'
            elif path == '/collector-monitor-model.mjs':
                data = (Path(__file__).parent / 'src/collector-monitor-model.mjs').read_bytes()
                mime = 'text/javascript; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass
    return Monitor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    print(f'Charlie collection monitor: http://127.0.0.1:{args.port}', flush=True)
    def schedule_loop():
        while True:
            c = None
            try:
                c = Collector(args.state)
                RefreshManager(c).due()
            except Exception as exc:
                print(f'Refresh scheduler needs attention: {type(exc).__name__}', flush=True)
            finally:
                if c: c.db.close()
            time.sleep(30)
    threading.Thread(target=schedule_loop, daemon=True).start()
    HTTPServer(('127.0.0.1', args.port), handler(args.state, args.port)).serve_forever()
