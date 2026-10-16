"""Synthetic edition validation/rendering; no production decisions or model calls."""
import copy
import unittest
import uuid

from onepager_corrections import prepare, preparation
from onepager_edition_contract import baseline, prepare_edition
from onepager_edition_evidence import pending_evidence, accepted_evidence
from research_review_brief import render
from tests.unit.test_onepager_accepted_render import decision
from tests.unit.test_onepager_edition_contract import edition_fixture
from stock_evidence import report_fields


def fixture(statement='Qualified FY2029 assumption <not a target>.'):
    run, assignment, body, records, docs, financials = edition_fixture()
    run['state']['report'] = {
        'summary': {'one_liner': 'Old claim', 'investment_thesis': ['Old claim'], 'earnings_direction': 'Old claim'},
        'debates': [{'evidence_for_market': 'Old claim', 'evidence_for_alternative': 'Old claim'}],
        'valuation': {'summary': 'Old claim', 'metrics': [{'historical_range': 'Old claim', 'peer_reference': 'Old claim'}]},
        'monitor': {'kpis': [{'current': 'Old claim', 'bull_threshold': 'Old claim', 'bear_threshold': 'Old claim', 'timing': 'Old claim'}]},
        'scenarios': [dict(name=name, **{key: 'Old claim' for key in ('business_conditions', 'eps_or_fcf', 'valuation_multiple', 'implied_value', 'upside_downside')}) for name in ('Bear', 'Base', 'Bull')],
    }
    source = run['sources'][0]
    source['coverage'] = {'mode': 'bounded_excerpts', 'limitation': 'Only selected passages reviewed.'}
    source['textExtraction'] = {'limitations': ['Charts not extracted.']}
    assignment['input'].update(since='2026-01-01', until='2026-10-15', horizon='FY2029',
        labPack=[{'documentId': '19', 'filename': source['filename'], 'sha256': source['originalHash']}],
        processingSourceContext={'19': {'role': 'supplemental', 'publishedDate': '2025-11-01', 'publisher': '<untrusted>', 'sourceKind': 'broker-report', 'referencedBySavedThesis': True}})
    fingerprint = preparation(run)['fingerprint']
    proposal = records[0]['proposal']
    records = []
    for path in report_fields(run['state']['report']):
        if path.endswith('/name'):
            continue
        correction = {key: proposal[key] for key in ('basis', 'reason', 'sourceId', 'excerpt', 'locator')}
        correction.update(requestId=str(uuid.uuid4()), path=path, fingerprint=fingerprint, statement=statement)
        ident, signature, prepared = prepare(run, correction, docs['fixture.txt'])
        records.append(dict(id=ident, run_id=run['id'], fingerprint=fingerprint, payload_hash=signature, proposal=prepared))
    body.update({key: value for key, value in baseline(run, assignment, financials).items() if key != 'ticker'})
    body['correctionIds'] = [row['id'] for row in records]
    record = prepare_edition(run, assignment, body, records, docs, financials)
    return run, assignment, record, records, docs, financials


class RecoveryBriefTests(unittest.TestCase):
    def test_developed_paragraph_is_rendered_whole_without_silent_truncation(self):
        paragraph = ('Synthetic recurring demand supports earnings only if retention and cash conversion persist. ' * 7).strip()
        args = fixture(paragraph)
        rendered = render(args[0], args[1]['input'], preview=pending_evidence(*args))
        self.assertIn(paragraph, rendered)
        self.assertIn('Pending editorial preview', rendered)

    def test_pending_status_and_scope_come_from_saved_edition(self):
        args = fixture(); before = copy.deepcopy(args)
        rendered = render(args[0], args[1]['input'], preview=pending_evidence(*args))
        self.assertIn('Pending editorial preview — not accepted', rendered)
        self.assertIn('1 original sources', rendered)
        self.assertIn('FY2029', rendered)
        self.assertLess(rendered.index('<tr><th>Bear'), rendered.index('<tr><th>Bull'))
        self.assertIn('<sup>[1]</sup>', rendered)
        self.assertIn('Qualified FY2029 assumption &lt;not a target&gt;.', rendered)
        self.assertIn('Discovery dates limit source search, not the longitudinal investment case.', rendered)
        for misleading in ('End-2027', 'eight selected', 'six monitoring', 'returns exclude dividends', 'Investor-accepted'):
            self.assertNotIn(misleading, rendered)
        self.assertEqual(args, before)

    def test_validated_acceptance_renders_with_decision_without_pending_status(self):
        args = fixture(); before = copy.deepcopy(args)
        context = accepted_evidence(*args, decision=decision(args[2]))
        rendered = render(args[0], args[1]['input'], preview=context)
        self.assertIn('Investor-accepted editorial edition', rendered)
        self.assertIn(context.decision_id, rendered)
        self.assertIn(context.decided_at, rendered)
        self.assertIn('Acceptance does not certify source accuracy or approve a thesis.', rendered)
        self.assertNotIn('Pending', rendered)
        self.assertEqual(args, before)

    def test_frozen_metadata_and_review_limits_are_visible_and_escaped(self):
        args = fixture()
        rendered = render(args[0], args[1]['input'], preview=pending_evidence(*args))
        for text in ('Supplemental history', '2025-11-01', '&lt;untrusted&gt;', 'incorporation and current validity are not established', 'Only selected passages reviewed.', 'Charts not extracted.', args[0]['sources'][0]['originalHash']):
            self.assertIn(text, rendered)
        self.assertNotIn('<untrusted>', rendered)

    def test_missing_context_and_tampering_fail_closed(self):
        args = fixture()
        for preview in (None, {'status': 'accepted'}):
            with self.assertRaises(ValueError): render(args[0], args[1]['input'], preview=preview)
        context = pending_evidence(*args)
        context.claims['/summary/one_liner']['reviewOrigin'] = 'investor'
        with self.assertRaises(ValueError): render(args[0], args[1]['input'], preview=context)
        context = pending_evidence(*args)
        with self.assertRaises(ValueError): render(args[0], {**args[1]['input'], 'horizon': 'changed'}, preview=context)

    def test_incomplete_editorial_selection_is_not_silently_substituted(self):
        from tests.unit.test_onepager_edition_evidence import fixture as incomplete
        args = incomplete()
        with self.assertRaisesRegex(ValueError, 'explicit edition correction'):
            render(args[0], args[1]['input'], preview=pending_evidence(*args))

    def test_stale_or_restricted_original_cannot_be_rendered_as_accepted(self):
        for change in ('bytes', 'usage'):
            args = fixture()
            if change == 'bytes': args[4]['fixture.txt']['file_data'] = 'Y2hhbmdlZA=='
            else: args[4]['fixture.txt']['metadata']['usage'] = 'reference_only'
            with self.assertRaises(ValueError): accepted_evidence(*args, decision=decision(args[2]))
