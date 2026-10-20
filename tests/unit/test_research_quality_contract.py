"""Offline evidence fidelity and rewrite identity regressions; no model calls."""
import copy
import unittest
from research_source_budget import complete_sources, FULL_TEXT_BYTE_LIMIT, fit_sources
from research_package_rewrite import specification, wire_spec, validate, prompt
from research_writing import guidance
from tests.unit.test_research_source_budget import source


class QualityContractTests(unittest.TestCase):
    def test_middle_evidence_survives_pack_above_legacy_limit(self):
        pack=[source(100000,'a'),source(100000,'b')]
        pack[0]['text']=pack[0]['text'][:70000]+'MATERIAL CONTRARY EVIDENCE'+pack[0]['text'][70025:]
        before=copy.deepcopy(pack)
        result=complete_sources(pack)
        self.assertEqual(result,before)
        self.assertIn('MATERIAL CONTRARY EVIDENCE',result[0]['text'])
        self.assertNotIn('MATERIAL CONTRARY EVIDENCE',fit_sources(pack)[0]['text'])
        result[0]['text']='changed'
        self.assertEqual(pack,before)

    def test_over_budget_or_clipped_pack_stops_without_mutation(self):
        for pack in [[source(FULL_TEXT_BYTE_LIMIT+1,'a')],fit_sources([source(100000,'a'),source(100000,'b')])]:
            before=copy.deepcopy(pack)
            with self.assertRaises(ValueError):complete_sources(pack)
            self.assertEqual(pack,before)
        with self.assertRaises(ValueError):complete_sources([{'text':'漢'*(FULL_TEXT_BYTE_LIMIT//3+1)}])

    def test_abt_size_package_roundtrip_and_identity_rejection(self):
        body={'writingPolicy':'synthesis-v2','instructions':'Review the investment argument',
              'fields':[{'id':f'original-uuid-{i}','label':f'Section {i}','text':'Existing text','limit':1600} for i in range(85)]}
        spec=specification(body);before=copy.deepcopy(spec)
        wire=wire_spec(spec)
        raw={'fields':[{'id':f['id'],'text':'Updated '+f['label']} for f in reversed(wire['fields'])]}
        result=validate(raw,spec)
        self.assertEqual(result['original-uuid-84'],'Updated Section 84')
        self.assertEqual(spec,before)
        raw['fields'][0]['id']='field-999'
        with self.assertRaises(ValueError):validate(raw,spec)
        raw['fields'][0]['id']='field-1'
        with self.assertRaises(ValueError):validate(raw,spec)

    def test_legacy_ids_and_policy_do_not_change(self):
        spec=specification({'writingPolicy':'synthesis-v1','instructions':'Revise',
                            'fields':[{'id':'legacy-uuid','label':'Summary','text':'Original','limit':1600}]})
        self.assertEqual(wire_spec(spec),spec)
        self.assertEqual(validate({'fields':[{'id':'legacy-uuid','text':'Kept'}]},spec),{'legacy-uuid':'Kept'})
        self.assertNotIn('INVESTMENT JUDGMENT STANDARD',prompt(spec,[],{}))
        self.assertIn('does NOT establish expected return',guidance('synthesis-v2'))


    def test_rewrite_reextracts_verified_originals_without_mutating_parent(self):
        import base64, hashlib
        from research_source_budget import rewrite_sources
        raw=b'original bytes';sha=hashlib.sha256(raw).hexdigest()
        old={'id':'stable-source','filename':'original.txt','originalHash':sha,
             'text':'old clipped text','selectionContext':{'role':'supplemental'}}
        parent={'sources':[old],'input':{'hashes':{'original.txt':sha}}}
        document={'filename':'original.txt','file_data':base64.b64encode(raw).decode()}
        before=copy.deepcopy(parent)
        fresh=rewrite_sources(parent,[document],lambda d:'Full returned text including middle evidence')
        self.assertEqual(fresh[0]['id'],old['id'])
        self.assertEqual(fresh[0]['coverage']['omittedCharacters'],0)
        self.assertEqual(fresh[0]['selectionContext'],old['selectionContext'])
        self.assertEqual(parent,before)
        with self.assertRaises(ValueError):
            rewrite_sources(parent,[{**document,'file_data':base64.b64encode(b'changed').decode()}],lambda d:self.fail('Extracted unverified source'))
