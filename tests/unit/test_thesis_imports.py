import copy
import json
import unittest
from thesis_imports import normalize, candidate, digest, row_hash, SCHEMA


def sample(ticker='TEST'):
    return {'schema': SCHEMA, 'ticker': ticker, 'companyName': 'Synthetic company',
        'analysis': {'ticker': ticker, 'company': 'Synthetic company',
            'thesis': {'summary': 'A conditional investment hypothesis.', 'pillars': [
                {'id': 'p1', 'title': 'Revenue quality', 'description': 'Retention must improve.', 'sources': [
                    {'sourceId': 's1', 'filename': 'synthetic.pdf', 'pdfPages': [2], 'sha256': 'a'*64}]}]},
            'signposts': [{'id':'s1', 'metric': 'Retention', 'target': 'Above 90%', 'timeframe': 'Next quarter'}],
            'threats': [{'id':'r1', 'threat': 'Customer losses', 'triggerPoints': 'Retention below 80%'}],
            'conclusion': 'Medium conviction', 'documentHistory': [{'filename': 'synthetic.pdf'}]},
        'sourceRegister': [{'id':'s1', 'name':'synthetic.pdf','pages':4,'sha256':'a'*64}], 'baseline':{}}


class DraftValidationTests(unittest.TestCase):
    def test_supported_formats_and_no_trusted_approval(self):
        d=sample();d['schema']='charlie.external-thesis-draft.proposed.v1';d['status']='approved'
        d['analysis']['history']=[{'fake':'history'}]
        d['analysis']['documentHistory'][0]['stored']=True
        clean=normalize(d)
        self.assertEqual(clean['schema'],SCHEMA)
        self.assertNotIn('status',clean)
        self.assertNotIn('history',clean['analysis'])
        self.assertNotIn('stored',clean['analysis']['documentHistory'][0])
        self.assertEqual(clean['analysis']['thesis']['pillars'][0]['sources'][0]['pdfPages'],[2])

    def test_rejects_wrong_company_malformed_fields_and_broken_sources(self):
        mutations=[lambda d:d.update(ticker='OTHER'),lambda d:d.update(companyName='Wrong'),
            lambda d:d['analysis'].update(thesis=[]),lambda d:d['analysis']['thesis'].update(summary={}),
            lambda d:d['analysis']['signposts'][0].update(target={}),
            lambda d:d['analysis']['threats'][0].update(triggerPoints=[]),
            lambda d:d['analysis']['thesis']['pillars'].append(copy.deepcopy(d['analysis']['thesis']['pillars'][0])),
            lambda d:d['analysis']['thesis']['pillars'][0]['sources'][0].update(sourceId='missing'),
            lambda d:d['analysis']['thesis']['pillars'][0]['sources'][0].update(sha256='b'*64),
            lambda d:d['analysis']['thesis']['pillars'][0]['sources'][0].update(pdfPages=[5]),
            lambda d:d.update(provenance={'value':float('nan')}),
            lambda d:d.update(provenance={'note':'\x00'}),lambda d:d.update(provenance={'note':'\ud800'}),
            lambda d:d.update(baseline={'expectedNoExistingThesis':'false'}),lambda d:d.update(schema='unknown'),lambda d:d.update(baseline=[])]
        for change in mutations:
            d=sample();change(d)
            with self.subTest(change=change), self.assertRaises(ValueError):normalize(d)

    def test_preserves_prior_state_without_trusting_imported_history(self):
        p=normalize(sample());old=copy.deepcopy(p['analysis']);old['thesis']['summary']='Old thesis'
        old['documentHistory']=[{'filename':'older.pdf'}];old['customNote']='Keep me';old['history']=[]
        result=candidate(old,p,'draft-id')
        self.assertEqual(old['thesis']['summary'],'Old thesis')
        self.assertEqual(result['customNote'],'Keep me')
        self.assertEqual(result['history'][0]['thesis']['summary'],'Old thesis')
        self.assertEqual(len(result['documentHistory']),2)
        self.assertEqual(result['externalDraft']['execution'],'external')

    def test_version_identity_covers_timestamp_company_and_content(self):
        row={'analysis':{},'company':'One','updated_at':'date1'}
        self.assertNotEqual(row_hash(row),row_hash({**row,'updated_at':'date2'}))
        self.assertNotEqual(row_hash(row),row_hash({**row,'company':'Two'}))
        self.assertNotEqual(row_hash(None),row_hash(row))
        self.assertEqual(digest({'a':1,'b':2}),digest({'b':2,'a':1}))
