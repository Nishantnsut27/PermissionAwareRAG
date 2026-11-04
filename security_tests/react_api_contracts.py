"""Read-only migration contract checks; generation and provider access are mocked."""
import sys
import unittest
import socket
import threading
import time
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from answering.src import api, config, service
from security_tests.phase1_identity_boundary import api_server
from evaluation import reports

ORIGIN = 'https://migration-review.vercel.app'


class MigrationContracts(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(config, 'API_ALLOWED_ORIGINS', (ORIGIN,)))
        self.enterContext(patch.object(config, 'API_DEMO_MODE', True))
        self.enterContext(patch.object(config, 'API_PUBLIC_DEMO_MODE', False))
        self.enterContext(patch.dict(api._demo_sessions, {}, clear=True))
        self.client = requests.Session()
        self.client.trust_env = False
        self.addCleanup(self.client.close)
        self.base = self.enterContext(api_server())

    def test_allowed_preflight_and_error_headers(self):
        response = self.client.options(self.base + '/demo/query/stream', headers={
            'Origin': ORIGIN, 'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'authorization, content-type, accept'}, timeout=5)
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.headers.get('Access-Control-Allow-Origin'), ORIGIN)
        for path in ('/demo/me', '/not-found'):
            response = self.client.get(self.base + path, headers={'Origin': ORIGIN}, timeout=5)
            self.assertIn(response.status_code, (401, 404))
            self.assertEqual(response.headers.get('Access-Control-Allow-Origin'), ORIGIN)
            self.assertEqual(response.headers.get('Cache-Control'), 'no-store')

    def test_disallowed_origins_and_preflight_fields(self):
        response = self.client.get(self.base + '/demo/users', headers={'Origin': 'https://untrusted.example'}, timeout=5)
        self.assertEqual(response.status_code, 403)
        self.assertNotIn('Access-Control-Allow-Origin', response.headers)
        for method, headers in [('DELETE', 'authorization'), ('POST', 'x-user-id')]:
            response = self.client.options(self.base + '/demo/query/stream', headers={
                'Origin': ORIGIN, 'Access-Control-Request-Method': method,
                'Access-Control-Request-Headers': headers}, timeout=5)
            self.assertEqual(response.status_code, 403)

    def test_session_restoration_is_server_derived(self):
        response = self.client.post(self.base + '/demo/login', json={'user_id': 'U-001', 'password': 'aditya@login'}, timeout=5)
        self.assertEqual(response.status_code, 200)
        token = response.json()['access_token']
        original = api._demo_sessions[token]
        restored = self.client.get(self.base + '/demo/me?user_id=U-005', headers={'Origin': ORIGIN, 'Authorization': f'Bearer {token}'}, timeout=5)
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(restored.json()['user']['user_id'], 'U-001')
        self.assertEqual(api._demo_sessions[token], original)
        api._demo_sessions[token] = ('U-001', 0)
        for authorization in (f'Bearer {token}', 'Bearer invalid', ''):
            response = self.client.get(self.base + '/demo/me', headers={'Origin': ORIGIN, 'Authorization': authorization}, timeout=5)
            self.assertEqual(response.status_code, 401)

    def test_stream_cors_and_principal(self):
        token, _ = api._issue_demo_session('U-001')
        event = {'type': 'done', 'outcome': 'refusal', 'completed': True, 'answer': '', 'sources': [], 'stats': {}, 'effective_sellers': []}
        with patch.object(service, 'answer_query_stream', return_value=iter([event])) as generate:
            response = self.client.post(self.base + '/demo/query/stream', headers={'Origin': ORIGIN, 'Authorization': f'Bearer {token}'}, json={'query': 'review'}, timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get('Access-Control-Allow-Origin'), ORIGIN)
        self.assertIn('text/event-stream', response.headers['Content-Type'])
        self.assertIn('"type": "done"', response.text)
        self.assertEqual(generate.call_args.kwargs['principal'].user.user_id, 'U-001')

    def test_evaluation_remains_public_summary_only(self):
        response = self.client.get(self.base + '/evaluation', headers={'Origin': ORIGIN}, timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertTrue({'results', 'matrix', 'summary'}.isdisjoint(response.json()))
        token, _ = api._issue_demo_session('U-001')
        with patch.object(config, 'EVALUATION_REPORTS_ENABLED', True):
            response = self.client.get(self.base + '/evaluation/matrix', headers={'Authorization': f'Bearer {token}'}, timeout=5)
        self.assertEqual(response.status_code, 403)

    def test_every_account_can_read_same_summary_without_providers(self):
        with patch.object(service, 'answer_query', side_effect=AssertionError('provider invoked')), \
             patch.object(service, 'answer_query_stream', side_effect=AssertionError('provider invoked')):
            bodies = []
            for user_id in service.USERS:
                token, _ = api._issue_demo_session(user_id)
                response = self.client.get(self.base + '/evaluation', headers={'Authorization': f'Bearer {token}'}, timeout=5)
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertIn('measurements', body)
                for forbidden in ('question', 'answer', 'source_provenance', 'security_trace', 'document_id', 'judge_notes'):
                    self.assertNotIn('"' + forbidden + '"', json.dumps(body))
                bodies.append(body)
            self.assertTrue(all(body == bodies[0] for body in bodies))


class OverloadContracts(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(config, 'API_ALLOWED_ORIGINS', (ORIGIN,)))
        self.enterContext(patch.object(config, 'API_MAX_CONCURRENT_REQUESTS', 1))
        self.server = api.BoundedHTTPServer(('127.0.0.1', 0), api.Handler)
        self.server._admission.acquire()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def connect(self):
        client = socket.create_connection(self.server.server_address, timeout=2)
        self.addCleanup(client.close)
        return client

    def test_overload_json_cors_without_auth_body_or_generation(self):
        with patch.object(api.Handler, '_authenticate', side_effect=AssertionError('authentication')), \
             patch.object(api.Handler, '_read_json_body', side_effect=AssertionError('body')), \
             patch.object(service, 'answer_query', side_effect=AssertionError('generation')):
            for origin in (ORIGIN, 'https://untrusted.example', None):
                client = self.connect()
                head = 'POST /demo/query HTTP/1.1\r\nHost: localhost\r\nContent-Length: 999999\r\n'
                if origin: head += 'Origin: ' + origin + '\r\n'
                client.sendall((head + '\r\n').encode())
                response = b''
                while block := client.recv(4096): response += block
                self.assertIn(b'503 Service Unavailable', response)
                self.assertIn(b'Retry-After: 1', response)
                self.assertEqual(json.loads(response.split(b'\r\n\r\n')[1])['error'], 'overloaded')
                self.assertEqual(b'Access-Control-Allow-Origin:' in response, origin == ORIGIN)

    def test_two_rejection_slots_bound_slow_headers_and_close_excess(self):
        first, second = self.connect(), self.connect()
        first.sendall(b'GET / HTTP/1.1\r\n')
        second.sendall(b'GET / HTTP/1.1\r\n')
        deadline = time.monotonic() + .5
        while self.server._rejections._value and time.monotonic() < deadline:
            time.sleep(.005)
        third = self.connect()
        self.assertEqual(third.recv(1), b'')
        started = time.monotonic()
        self.assertEqual(first.recv(1), b'')
        self.assertEqual(second.recv(1), b'')
        self.assertLess(time.monotonic() - started, 1.5)
        deadline = time.monotonic() + .5
        while self.server._rejections._value < 2 and time.monotonic() < deadline: time.sleep(.005)
        self.assertEqual(self.server._rejections._value, 2)

    def test_oversized_headers_close_without_a_handler(self):
        client = self.connect()
        client.sendall(b'GET / HTTP/1.1\r\nX-Large: ' + b'a' * (64 * 1024))
        try:
            self.assertEqual(client.recv(1024), b'')
        except ConnectionResetError:
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
