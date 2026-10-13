import copy
import unittest
from onepager_corrections import digest
from onepager_edition_evidence import accepted_evidence
from onepager_readiness import readiness
from research_onepager import render as brief
from research_infographic import render as illustrated
from stock_evidence import valid_citations
from tests.unit.test_onepager_edition_evidence import fixture


def decision(record):
    return dict(requestId='b998e805-0bd4-49ce-aafa-2af123e1aace',editionId=record['id'],
                action='accept',decidedAt='2026-10-12T14:00:00Z',editionDigest=digest(record))


class AcceptedRenderTests(unittest.TestCase):
    def test_both_layouts_retain_explicit_human_provenance_and_original(self):
        args=fixture();before=copy.deepcopy(args);context=accepted_evidence(*args,decision=decision(args[2]))
        for renderer in (lambda **kw:brief(args[0],args[1]['input'],**kw),lambda **kw:illustrated(args[0],args[1]['input'],args[5],**kw)):
            original=renderer();rendered=renderer(preview=context)
            self.assertIn('Proposed claim',rendered)
            self.assertIn('Investor-reviewed editorial wording',rendered)
            self.assertIn('Investor-accepted editorial edition',rendered)
            self.assertIn(context.decided_at,rendered)
            self.assertNotIn('Pending editorial proposal',rendered)
            self.assertEqual(original,renderer())
        self.assertEqual(args,before)
        self.assertNotIn('/summary/earnings_direction',valid_citations(context.run))
        row=readiness(args[0],{'view':True},preview=context)[2]
        self.assertEqual(row['pendingPaths'],[]);self.assertEqual(row['supportedPaths'],[])
        self.assertEqual(row['investorPaths'],['/summary/earnings_direction'])

    def test_accepted_print_header_does_not_revert_to_pending(self):
        args=fixture();context=accepted_evidence(*args,decision=decision(args[2]))
        rendered=illustrated(args[0],args[1]['input'],args[5],preview=context)
        self.assertIn('Charlie / Investor-accepted editorial edition',rendered)
        self.assertNotIn('Charlie / Pending review - not accepted',rendered)
        self.assertIn('counter(page) " of " counter(pages)',rendered)

    def test_no_pending_rejected_changed_or_restricted_export(self):
        for change in ('action','digest','edition','source','financials'):
            args=fixture();d=decision(args[2])
            if change=='action':d['action']='reject'
            if change=='digest':d['editionDigest']='bad'
            if change=='edition':d['editionId']='c998e805-0bd4-49ce-aafa-2af123e1aace'
            if change=='source':args[4]['fixture.txt']['metadata']['usage']='reference_only'
            if change=='financials':args[5]['price']=101
            with self.subTest(change=change),self.assertRaises(ValueError):accepted_evidence(*args,decision=d)

    def test_mutated_context_and_long_qualification_fail_closed(self):
        args=fixture(statement='Qualification '+('x'*300)+' subject to risk.')
        context=accepted_evidence(*args,decision=decision(args[2]))
        self.assertIn('data-readiness="over_budget"',illustrated(args[0],args[1]['input'],args[5],preview=context))
        self.assertIn('subject to risk.',brief(args[0],args[1]['input'],preview=context))
        context.claims['/summary/earnings_direction']['statement']='Changed'
        with self.assertRaises(ValueError):brief(args[0],args[1]['input'],preview=context)

    def test_only_verified_investor_session_supplies_identity(self):
        import ast
        from pathlib import Path
        from types import SimpleNamespace
        tree=ast.parse(Path('app_v3.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_edition_investor_identity')
        for password,header,expected in [('configured','Bearer valid','charlie-password-investor'),('configured','Bearer invalid',None),('configured','ApiKey machine',None),('', 'Bearer valid',None),('configured','',None)]:
            scope={'CHARLIE_PASSWORD':password,'request':SimpleNamespace(headers={'Authorization':header}),'_verify_session_token':lambda t:t=='valid'}
            exec(compile(ast.Module(body=[function],type_ignores=[]),'auth-fixture','exec'),scope)
            self.assertEqual(scope['_edition_investor_identity'](),expected)
