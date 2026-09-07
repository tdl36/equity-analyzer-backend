"""Read-only, loopback-only monitor for Charlie's local source collection ledger."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit
from charlie_collector import DEFAULT_STATE, now


def snapshot(state):
    path = Path(state) / 'ledger.sqlite3'
    if not path.exists():
        return {'checked': now(), 'runs': []}
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        runs = []
        for run in db.execute('SELECT * FROM runs ORDER BY created DESC LIMIT 20'):
            item = dict(run)
            for table in ('tasks', 'documents', 'observations'):
                item[table] = [dict(r) for r in db.execute(f'SELECT * FROM {table} WHERE run=?', (run['id'],))]
            for doc in item['documents']:
                doc.pop('staged', None)
                doc['destination'] = Path(doc['destination']).name if doc['destination'] else None
            runs.append(item)
        return {'checked': now(), 'runs': runs}
    finally:
        db.close()


def handler(state, port):
    class Monitor(BaseHTTPRequestHandler):
        def do_GET(self):
            # Reject rebinding/foreign-origin access. No CORS or write routes.
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
                    data = json.dumps(snapshot(state)).encode()
                except sqlite3.Error:
                    self.send_error(503, 'Collection ledger temporarily unavailable')
                    return
                mime = 'application/json'
            elif path == '/':
                data = (Path(__file__).parent / 'src/collector-monitor.html').read_bytes()
                mime = 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
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
    HTTPServer(('127.0.0.1', args.port), handler(args.state, args.port)).serve_forever()
