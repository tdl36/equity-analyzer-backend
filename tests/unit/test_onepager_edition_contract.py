import copy
import unittest
from onepager_corrections import prepare, preparation
from onepager_edition_contract import baseline, prepare_edition
from tests.unit.test_onepager_corrections import fixture, CID, RID

AID = 'f9e363e4-597e-4394-b9d4-36d83def935f'
EID = 'a998e805-0bd4-49ce-aafa-2af123e1aace'


def edition_fixture():
    run, doc, correction = fixture()
    ident, signature, proposal = prepare(run, correction, doc)
    assignment = {'id': AID, 'ticker': 'TEST', 'input': {'ticker': 'TEST'},
                  'result': {'reportId': RID, 'visualFundamentals': {'asOf': '2026-10-12', 'price': 100}}}
    fundamentals = {'price': 100, 'history': [{'year': 2024, 'eps': 5}]}
    expected = baseline(run, assignment, fundamentals)
    body = {k: v for k, v in expected.items() if k != 'ticker'}
    body.update(requestId=EID, correctionIds=[CID])
    records = [{'id': ident, 'run_id': RID, 'fingerprint': correction['fingerprint'],
                'payload_hash': signature, 'proposal': proposal}]
    return run, assignment, body, records, {doc['filename']: doc}, fundamentals


class EditionContractTests(unittest.TestCase):
    def test_pending_identity_retains_originals_and_never_forges_approval(self):
        args = edition_fixture(); before = copy.deepcopy(args)
        result = prepare_edition(*args)
        self.assertEqual(args, before)
        self.assertEqual(result['status'], 'pending')
        self.assertEqual(result['claims'][0]['proposal']['review'], 'pending')
        self.assertNotIn('reviewOrigin', str(result))
        self.assertEqual(result, prepare_edition(*args))
        self.assertNotIn('Synthetic source', str(baseline(args[0], args[1], args[5])))

    def test_stale_report_assignment_fundamentals_input_and_parent(self):
        for key in ['report', 'assignment', 'ticker', 'fundamentals', 'input', 'parent', 'snapshot']:
            args = edition_fixture(); run, assignment, *_ = args
            if key == 'report': run['state']['report']['changed'] = 'New'
            if key == 'assignment': assignment['id'] = EID
            if key == 'ticker': assignment['ticker'] = 'OTHER'
            if key == 'fundamentals': assignment['result']['visualFundamentals']['price'] = 101
            if key == 'snapshot': args[5]['history'][0]['eps'] = 6
            if key == 'input': assignment['input']['outputs'] = ['changed']
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_edition(*args, parent_id=EID if key == 'parent' else None)

    def test_unknown_flags_bounds_duplicate_and_unrelated_selection(self):
        for patch in [{'reviewOrigin': 'investor'}, {'status': 'accepted'},
                      {'correctionIds': []}, {'correctionIds': [CID]*21},
                      {'correctionIds': [CID, CID]}, {'correctionIds': [EID]},
                      {'correctionIds': [None]}]:
            args = edition_fixture(); args[2].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError): prepare_edition(*args)

    def test_changed_saved_provenance_is_not_reinterpreted(self):
        for key, value in [('before', 'different'), ('sha256', '0'*64),
                           ('review', 'supported'), ('passageMatched', False),
                           ('statement', 'A new claim'), ('filename', 'other.txt')]:
            args = edition_fixture(); args[3][0]['proposal'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): prepare_edition(*args)
        for patch in [{'run_id': EID}, {'fingerprint': 'changed'}, {'payload_hash': 'changed'}]:
            args = edition_fixture(); args[3][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError): prepare_edition(*args)

    def test_current_original_and_restrictions_are_rechecked(self):
        for key in ['missing', 'bytes', 'policy']:
            args = edition_fixture(); docs = args[4]
            if key == 'missing': docs.clear()
            elif key == 'bytes': docs['fixture.txt']['file_data'] = 'Y2hhbmdlZA=='
            else: docs['fixture.txt']['metadata'] = {'usage': 'reference_only'}
            with self.subTest(key=key), self.assertRaises(ValueError): prepare_edition(*args)

    def test_conflicting_claims_are_rejected_and_order_is_bound(self):
        args = edition_fixture(); row = copy.deepcopy(args[3][0]); row['id'] = EID
        args[3].append(row); args[2]['correctionIds'].append(EID)
        with self.assertRaisesRegex(ValueError, 'one correction'): prepare_edition(*args)
        # Duplicate server records are ambiguous even when their payloads agree.
        args = edition_fixture(); args[3].append(copy.deepcopy(args[3][0]))
        with self.assertRaisesRegex(ValueError, 'Ambiguous'): prepare_edition(*args)

    def test_missing_financial_snapshot_cannot_be_silently_refetched(self):
        args = edition_fixture(); args[5].clear()
        with self.assertRaisesRegex(ValueError, 'snapshot'): prepare_edition(*args)

    def test_distinct_claim_order_changes_selection_without_mutating_inputs(self):
        args = list(edition_fixture()); run, assignment, body, records, docs, financials = args
        run['state']['report']['summary']['other'] = 'Other saved claim'
        fingerprint = preparation(run)['fingerprint']
        records.clear()
        for ident, path in [(CID, '/summary/earnings_direction'), (EID, '/summary/other')]:
            correction = fixture()[2]
            correction.update(requestId=ident, path=path, fingerprint=fingerprint)
            _, signature, proposal = prepare(run, correction, docs['fixture.txt'])
            records.append({'id': ident, 'run_id': RID, 'fingerprint': fingerprint,
                            'payload_hash': signature, 'proposal': proposal})
        body.update(fingerprint=fingerprint, correctionIds=[CID, EID])
        first = prepare_edition(*args)
        body['correctionIds'].reverse()
        second = prepare_edition(*args)
        self.assertNotEqual(first['selectionDigest'], second['selectionDigest'])
        self.assertEqual([x['correctionId'] for x in second['claims']], [EID, CID])

    def test_complete_repair_selection_is_bounded_at_sixty(self):
        args = list(edition_fixture())
        run, assignment, body, records, docs, financials = args
        import uuid
        run['state']['report']['summary'].update({f'field{i}':'Old claim' for i in range(61)})
        fingerprint = preparation(run)['fingerprint']
        records.clear(); ids=[]
        for i in range(61):
            ident=str(uuid.uuid4()); ids.append(ident)
            correction=fixture()[2]
            correction.update(requestId=ident,path=f'/summary/field{i}',fingerprint=fingerprint)
            _, signature, proposal=prepare(run,correction,docs['fixture.txt'])
            records.append({'id':ident,'run_id':RID,'fingerprint':fingerprint,'payload_hash':signature,'proposal':proposal})
        body.update(fingerprint=fingerprint,correctionIds=ids[:60])
        self.assertEqual(len(prepare_edition(*args)['claims']),60)
        body['correctionIds']=ids
        with self.assertRaisesRegex(ValueError,'sixty'): prepare_edition(*args)
        body['correctionIds']=ids[:60]
        records[-2]['proposal']['sha256']='0'*64
        with self.assertRaises(ValueError): prepare_edition(*args)
