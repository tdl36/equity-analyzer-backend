import json
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
import requests
from collector_status import handler


class MonitorHTTPTests(unittest.TestCase):
    def test_mutations_require_same_origin_and_nonce(self):
        with tempfile.TemporaryDirectory() as tmp:
            server=HTTPServer(('127.0.0.1',0),handler(Path(tmp),0))
            port=server.server_port
            server.RequestHandlerClass=handler(Path(tmp),port)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                base=f'http://127.0.0.1:{port}'
                data=requests.get(base+'/api/status').json()
                self.assertIn('refreshToken',data)
                self.assertEqual(requests.post(base+'/api/refresh/trigger',json={'ticker':'MDT'}).status_code,403)
                self.assertEqual(requests.post(base+'/api/refresh/trigger',json={'ticker':'MDT'},headers={'Origin':'https://evil.invalid','X-Refresh-Token':data['refreshToken']}).status_code,403)
                self.assertEqual(requests.post(base+'/api/refresh/trigger',json={'ticker':'MDT'},headers={'Origin':base,'X-Refresh-Token':data['refreshToken']}).status_code,400)
                self.assertEqual(requests.get(base+'/api/status',headers={'Host':'evil.invalid'}).status_code,403)
            finally:
                server.shutdown();server.server_close();thread.join()
