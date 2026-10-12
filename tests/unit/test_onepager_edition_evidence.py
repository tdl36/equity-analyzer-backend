import copy
import unittest
from onepager_corrections import prepare, preparation
from onepager_edition_contract import baseline, prepare_edition
from onepager_edition_evidence import pending_evidence
from stock_evidence import valid_citations
from tests.unit.test_onepager_edition_contract import edition_fixture


def fixture(path='/summary/earnings_direction', report=None):
    args = list(edition_fixture())
    run, assignment, body, rows, docs, financials = args
    if report is not None: run['state']['report'] = report
    # Include original model review to prove editing does not inherit support.
    run['state']['citations'] = {path: {'statement':'Old claim', 'review':'supported',
        'passageMatched':True,'evidence':[{'sourceId':'s1','matched':True}]}}
    fingerprint = preparation(run)['fingerprint']
    p = rows[0]['proposal']
    proposal_body = {key:p[key] for key in ('statement','basis','reason','sourceId','excerpt','locator')}
    proposal_body.update(requestId=rows[0]['id'], path=path, fingerprint=fingerprint)
    ident, signature, proposal = prepare(run, proposal_body, docs['fixture.txt'])
    rows[0].update(fingerprint=fingerprint, payload_hash=signature, proposal=proposal)
    body.update({k:v for k,v in baseline(run,assignment,financials).items() if k!='ticker'})
    record = prepare_edition(*args)
    return run, assignment, record, rows, docs, financials


class PendingEvidenceTests(unittest.TestCase):
    def test_preview_is_separate_and_never_promotes_shared_evidence(self):
        args = fixture(); before = copy.deepcopy(args); run = args[0]
        preview = pending_evidence(*args)
        self.assertEqual(args, before)
        path = '/summary/earnings_direction'
        self.assertIn(path, valid_citations(run))
        self.assertNotIn(path, valid_citations(preview.run))
        self.assertEqual(preview.run['state']['citations'], run['state']['citations'])
        self.assertEqual(preview.claims[path]['reviewOrigin'], 'pending_editorial')
        self.assertNotIn('review', preview.claims[path])
        self.assertNotIn('passageMatched', preview.claims[path])
        self.assertIn('not accepted', preview.label)
        self.assertEqual(preview.run['state']['report']['summary']['earnings_direction'], 'Proposed claim')
        preview.run['sources'][0]['text'] = 'Local preview changed'
        self.assertEqual(args, before)

    def test_canonical_paths_preserve_arrays_and_escaped_keys(self):
        for path, report in [('/summary/a~1b~0c',{'summary':{'a/b~c':'Old claim'}}),
                             ('/scenarios/0/eps_or_fcf',{'scenarios':[{'eps_or_fcf':'Old claim'},{'eps_or_fcf':'Other case'}]})]:
            with self.subTest(path=path):
                args = fixture(path, report); preview = pending_evidence(*args)
                self.assertEqual(preview.claims[path]['statement'],'Proposed claim')
                if path.startswith('/scenarios'):
                    self.assertEqual(preview.run['state']['report']['scenarios'][1]['eps_or_fcf'],'Other case')
                else:self.assertEqual(preview.run['state']['report']['summary']['a/b~c'],'Proposed claim')

    def test_tampered_or_accepted_record_cannot_use_pending_adapter(self):
        for key, value in [('status','accepted'),('selectionDigest','forged'),('reviewOrigin','investor')]:
            args = fixture(); args[2][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): pending_evidence(*args)
        args = fixture(); args[2]['claims'][0]['proposal']['statement'] = 'Changed saved wording'
        with self.assertRaises(ValueError): pending_evidence(*args)

    def test_source_restriction_or_report_change_blocks_preview(self):
        args = fixture(); args[4]['fixture.txt']['metadata'] = {'usage':'reference_only'}
        with self.assertRaises(ValueError): pending_evidence(*args)
        args = fixture(); args[0]['state']['report']['summary']['earnings_direction'] = 'Changed original'
        with self.assertRaises(ValueError): pending_evidence(*args)
