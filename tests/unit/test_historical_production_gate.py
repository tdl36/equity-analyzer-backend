"""Production registration contract without importing the live Flask application."""
import ast
import hmac
from pathlib import Path
import unittest
from unittest.mock import Mock
from flask import Flask, request
from historical_valuation_api import create_blueprint


class HistoricalProductionGateTests(unittest.TestCase):
    def test_production_registration_is_readonly_and_has_no_provider_config(self):
        tree = ast.parse(Path('app_v3.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == 'historical_valuation_api']
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call.func.attr, 'create_blueprint')
        self.assertEqual(call.args[1].id, '_historical_evidence_authenticated')
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in call.keywords}, {'read_only': True})

    def test_machine_and_investor_auth_fail_closed(self):
        tree = ast.parse(Path('app_v3.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_historical_evidence_authenticated')
        investor = Mock(return_value=None)
        namespace = dict(hmac=hmac, request=request, CHARLIE_API_KEY='fixture-key',
                         _edition_investor_identity=investor)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'auth-fixture', 'exec'), namespace)
        app = Flask(__name__)
        for header, expected in [('', False), ('ApiKey wrong', False), ('ApiKey fixture-key', True)]:
            with app.test_request_context(headers={'Authorization':header}):
                self.assertEqual(namespace[fn.name](), expected)
        investor.return_value='fixture-investor'
        with app.test_request_context(headers={'Authorization':'Bearer fixture-session'}):
            self.assertTrue(namespace[fn.name]())
        investor.return_value=None; namespace['CHARLIE_API_KEY']=''
        with app.test_request_context(headers={'Authorization':'ApiKey '}):
            self.assertFalse(namespace[fn.name]())

    def test_readonly_rejects_all_mutations_before_db_or_config(self):
        db=Mock(side_effect=AssertionError('Database mutation attempted'))
        config=Mock(side_effect=AssertionError('Provider configuration accessed'))
        app=Flask(__name__);app.config['TESTING']=True
        authorized=[True]
        app.register_blueprint(create_blueprint(db,lambda:authorized[0],ingestion_config=config,read_only=True))
        client=app.test_client()
        base='/api/research/assignments/a/onepager-editions/e/'
        for path in ('historical-originals','historical-reviews/r'):
            response=client.post(base+path,json={'kind':'daily_prices'})
            self.assertEqual(response.status_code,405)
            self.assertEqual(response.headers['Cache-Control'],'private, no-store')
            authorized[0]=False
            self.assertEqual(client.post(base+path,json={}).status_code,403)
            authorized[0]=True
        db.assert_not_called();config.assert_not_called()
