import copy
import unittest
from onepager_edition_evidence import pending_evidence
from onepager_readiness import readiness, scenario_readiness
from research_onepager import render as brief
from research_infographic import render as illustrated
from stock_evidence import valid_citations
from tests.unit.test_onepager_edition_evidence import fixture


class PendingRenderTests(unittest.TestCase):
    def test_both_renderers_require_explicit_context_and_preserve_original(self):
        args = fixture(); run, assignment, _, _, _, financials = args
        before = copy.deepcopy(args); preview = pending_evidence(*args)
        for renderer in (lambda **kw:brief(run,assignment['input'],**kw),
                         lambda **kw:illustrated(run,assignment['input'],financials,**kw),
                         lambda **kw:brief(run,dict(assignment['input'],_visualFundamentals=financials),**kw)):
            original=renderer(); rendered=renderer(preview=preview)
            self.assertIn('Old claim',original);self.assertNotIn('Proposed claim',original)
            self.assertIn('Proposed claim',rendered)
            self.assertIn('Pending editorial proposal — not accepted',rendered)
            self.assertIn(preview.edition_id,rendered)
            self.assertEqual(original,renderer())
            with self.assertRaises(ValueError):renderer(preview={'reviewOrigin':'investor'})
        self.assertEqual(args,before)
        self.assertNotIn('/summary/earnings_direction',valid_citations(preview.run))

    def test_company_claim_has_valid_container_and_visible_pending_marker(self):
        args=fixture('/summary/one_liner', {'summary':{'one_liner':'Old claim'}})
        preview=pending_evidence(*args)
        rendered=illustrated(args[0],args[1]['input'],args[5],preview=preview)
        from bs4 import BeautifulSoup
        dom=BeautifulSoup(rendered,'html.parser')
        overview=dom.select_one('.overview')
        self.assertEqual([n.get('class') for n in overview.find_all(recursive=False)],
                         [['company-description'],['facts']])
        claim=overview.select_one('.company-description .claim')
        self.assertIn('Proposed claim',claim.get_text())
        marker=claim.select_one('.editorial-marker')
        self.assertEqual(marker.get_text(strip=True),'Pending proposal')
        self.assertEqual(marker['title'],'Pending editorial proposal — not accepted')
        self.assertIsNone(dom.select_one('p p'))
        self.assertIn('Pending editorial preview — not accepted', dom.get_text())

    def test_pending_readiness_is_not_supported_or_investor_accepted(self):
        args=fixture();preview=pending_evidence(*args)
        row=readiness(args[0],{'view':True},preview=preview)[2]
        self.assertEqual(row['status'],'rendered')
        self.assertEqual(row['supportedPaths'],[])
        self.assertEqual(row['pendingPaths'],['/summary/earnings_direction'])
        self.assertIn('not accepted',row['reason'])
        args=fixture('/scenarios/0/eps_or_fcf',{'scenarios':[{'eps_or_fcf':'Old claim'}, {'implied_value':'Other case'}]})
        rows=scenario_readiness(args[0],preview=pending_evidence(*args))['rows']
        self.assertEqual(next(f for f in rows[0]['fields'] if f['path'].endswith('/eps_or_fcf'))['status'],'pending')
        self.assertTrue(all(r['arithmetic']['status']=='unverified' for r in rows))

    def test_stale_original_tampered_context_and_changed_snapshot_are_rejected(self):
        for kind in ('report','context','financials','assignment'):
            args=fixture();run,assignment,*_=args;preview=pending_evidence(*args)
            if kind=='report':run['state']['report']['summary']['earnings_direction']='Changed'
            elif kind=='context':preview.claims['/summary/earnings_direction']['statement']='Forged'
            elif kind=='financials':args[5]['price']=101
            else:assignment['input']['until']='2099-01-01'
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                illustrated(run,assignment['input'],args[5],preview=preview)

    def test_long_qualified_claim_omitted_whole_and_diagnosed(self):
        statement='Long qualified proposal '+('x'*300)+' subject to uncertainty.'
        args=fixture(statement=statement);preview=pending_evidence(*args)
        rendered=illustrated(args[0],args[1]['input'],args[5],preview=preview)
        self.assertNotIn('Long qualified proposal',rendered)
        self.assertIn('data-readiness="over_budget"',rendered)
        self.assertIn('Pending proposal paths: /summary/earnings_direction',rendered)
        self.assertIn(statement,brief(args[0],args[1]['input'],preview=preview))

    def test_direct_headline_labels_and_html_escaping(self):
        args=fixture('/infographic/headline',{'infographic':{'headline':'Old claim'}},'<script>Proposed headline</script>')
        preview=pending_evidence(*args)
        for h in (brief(args[0],args[1]['input'],preview=preview),illustrated(args[0],args[1]['input'],args[5],preview=preview)):
            self.assertNotIn('<script>',h)
            self.assertIn('&lt;script&gt;Proposed headline&lt;/script&gt;',h)
            self.assertIn('Pending editorial proposal — not accepted',h.split('</h2>' if 'headline"' in h else '</h1>')[0])
