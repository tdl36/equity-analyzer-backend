import copy
import unittest
from onepager_corrections import prepare, preparation
from onepager_edition_contract import prepare_edition
from onepager_decision_contract import validate_decision
from tests.unit.test_onepager_edition_contract import edition_fixture, EID

DID = 'b998e805-0bd4-49ce-aafa-2af123e1aace'


def fixture(parent=None):
    run, assignment, selection, corrections, documents, financials = edition_fixture()
    selection['parentId'] = parent
    record = prepare_edition(run, assignment, selection, corrections, documents,
                             financials, parent_id=parent)
    body = {key: record[key] for key in ('assignmentId', 'runId', 'fingerprint',
            'fundamentalsDigest', 'parentId', 'correctionIds', 'selectionDigest')}
    body.update(requestId=DID, editionId=EID, action='accept')
    return run, assignment, record, body, corrections, documents, financials


class DecisionContractTests(unittest.TestCase):
    def test_intent_is_repeatable_without_mutation_or_acceptance(self):
        for action in ('accept', 'reject'):
            args = fixture(); args[3]['action'] = action
            before = copy.deepcopy(args)
            result = validate_decision(*args)
            self.assertEqual(result, validate_decision(*args))
            self.assertEqual(args, before)
            self.assertEqual(args[2]['status'], 'pending')
            self.assertEqual(result['action'], action)
            for key in ('status', 'claims', 'reviewOrigin', 'decidedAt', 'acceptedId'):
                self.assertNotIn(key, result)
            self.assertNotIn('Synthetic source', str(result))

    def test_exact_review_identity_and_unknown_flags(self):
        for key, value in [('requestId', None), ('editionId', DID),
                ('assignmentId', DID), ('runId', DID), ('fingerprint', 'changed'),
                ('fundamentalsDigest', 'changed'), ('parentId', DID),
                ('correctionIds', []), ('selectionDigest', 'changed'),
                ('action', 'approve'), ('reviewOrigin', 'investor'),
                ('decidedAt', '2026-10-12'), ('status', 'accepted')]:
            args = fixture(); args[3][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_decision(*args)

    def test_changed_parent_requires_new_preparation_for_both_actions(self):
        for action in ('accept', 'reject'):
            args = fixture(); args[3]['action'] = action
            with self.assertRaises(ValueError): validate_decision(*args, parent_id=DID)
            args = fixture(DID); args[3]['action'] = action
            self.assertEqual(validate_decision(*args, parent_id=DID)['parentId'], DID)
            with self.assertRaises(ValueError): validate_decision(*args, parent_id=EID)

    def test_stale_saved_inputs_and_tampering_fail(self):
        for kind in ('report', 'financials', 'assignment', 'correction', 'record', 'stopped', 'decided'):
            args = fixture()
            if kind == 'report': args[0]['state']['report']['changed'] = 'Changed'
            elif kind == 'financials': args[6]['price'] = 101
            elif kind == 'assignment': args[1]['input']['outputs'] = ['changed']
            elif kind == 'correction': args[4][0]['proposal']['statement'] = 'Changed'
            elif kind == 'record': args[2]['claims'][0]['proposalDigest'] = 'changed'
            elif kind == 'stopped': args[1]['status'] = 'stopped'
            else: args[2]['status'] = 'accepted'
            with self.subTest(kind=kind), self.assertRaises(ValueError): validate_decision(*args)

    def test_entire_pack_rechecked_even_for_unchanged_claims_and_rejection(self):
        for action in ('accept', 'reject'):
            for fault in ('missing', 'bytes', 'policy'):
                args = list(fixture()); run, assignment, record, body, corrections, docs, financials = args
                extra = {**run['sources'][0], 'id': 's2', 'filename': 'unchanged.txt'}
                run['sources'].append(extra)
                run['input']['hashes']['unchanged.txt'] = extra['originalHash']
                docs['unchanged.txt'] = {**docs['fixture.txt'], 'filename': 'unchanged.txt'}
                p = corrections[0]['proposal']
                cb = {k:p[k] for k in ('path','statement','basis','reason','sourceId','excerpt','locator')}
                cb.update(requestId=corrections[0]['id'], fingerprint=preparation(run)['fingerprint'])
                _, signature, proposal = prepare(run, cb, docs['fixture.txt'])
                corrections[0].update(fingerprint=cb['fingerprint'],payload_hash=signature,proposal=proposal)
                selection = {k:record[k] for k in ('assignmentId','runId','fundamentalsDigest','parentId','correctionIds')}
                selection.update(requestId=EID,fingerprint=cb['fingerprint'])
                args[2] = prepare_edition(run, assignment, selection, corrections, docs, financials)
                body.update(fingerprint=cb['fingerprint'],selectionDigest=args[2]['selectionDigest'],action=action)
                validate_decision(*args)
                if fault == 'missing': del docs['unchanged.txt']
                elif fault == 'bytes': docs['unchanged.txt']['file_data'] = 'Y2hhbmdlZA=='
                else: docs['unchanged.txt']['metadata'] = {'usage':'reference_only'}
                with self.subTest(action=action,fault=fault), self.assertRaises(ValueError): validate_decision(*args)

    def test_action_is_part_of_replay_digest(self):
        args = fixture(); first = validate_decision(*args)
        args[3]['action'] = 'reject'; second = validate_decision(*args)
        self.assertNotEqual(first['payloadDigest'], second['payloadDigest'])
        self.assertEqual(first['editionDigest'], second['editionDigest'])
