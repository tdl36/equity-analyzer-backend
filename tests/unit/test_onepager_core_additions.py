"""Synthetic pending content only; no investor approval or model calls."""
import copy
import unittest
import uuid
from onepager_core_fields import missing_fields,add_field
from onepager_corrections import prepare
from onepager_edition_contract import baseline,prepare_edition
from onepager_edition_evidence import pending_evidence
from onepager_readiness import readiness
from tests.unit.test_onepager_corrections import fixture,RID
from tests.unit.test_onepager_edition_contract import AID,EID

class CoreAdditionTests(unittest.TestCase):
    def test_missing_sections_become_explicit_pending_claims_without_rewriting_original(self):
        run,doc,body=fixture();run['state']['report']={}
        from onepager_corrections import preparation
        body['fingerprint']=preparation(run)['fingerprint'];before=copy.deepcopy(run)
        paths=['/summary/investment_thesis/2','/summary/one_liner','/summary/core_question','/summary/earnings_direction','/debates/1/alternative_view','/debates/1/market_view','/debates/1/resolution_signal']
        records=[]
        for path in paths:
            ident,signature,proposal=prepare(run,{**body,'requestId':str(uuid.uuid4()),'path':path},doc)
            self.assertIsNone(proposal['before'])
            records.append(dict(id=ident,run_id=RID,fingerprint=body['fingerprint'],payload_hash=signature,proposal=proposal))
        assignment={'id':AID,'ticker':'TEST','input':{},'result':{'reportId':RID,'visualFundamentals':{'price':100}}}
        financials={'price':100};pack=baseline(run,assignment,financials)
        selection={k:v for k,v in pack.items() if k!='ticker'};selection.update(requestId=EID,correctionIds=[r['id'] for r in records])
        edition=prepare_edition(run,assignment,selection,records,{doc['filename']:doc},financials)
        preview=pending_evidence(run,assignment,edition,records,{doc['filename']:doc},financials)
        self.assertEqual(run,before);self.assertEqual(len(preview.claims),len(paths))
        self.assertTrue(all(c['reviewOrigin']=='pending_editorial' for c in preview.claims.values()))
        self.assertEqual(preview.run['state']['report']['summary']['investment_thesis'][:2],[None,None])
        self.assertEqual(preview.run['state']['report']['debates'][0],{})
        rows=readiness(run,{'thesis':True,'debate':True,'view':True},preview=preview)
        self.assertTrue(all(r['pendingPaths'] for r in rows))
        run['state']['report']['summary']={'one_liner':'Changed'}
        with self.assertRaises(ValueError):prepare_edition(run,assignment,selection,records,{doc['filename']:doc},financials)

    def test_malformed_containers_and_unapproved_paths_are_never_replaced(self):
        for report,path in [({'summary':'bad'},'/summary/one_liner'),({'summary':{'investment_thesis':{}}},'/summary/investment_thesis/0'),({'summary':{'investment_thesis':[{}]}},'/summary/investment_thesis/0'),({'debates':[None]},'/debates/0/topic'),({'debates':{}},'/debates/0/topic'),({},'/summary/investment_thesis/3'),({},'/debates/2/topic'),({},'/summary/approval'),({},'/debates/00/topic')]:
            old=copy.deepcopy(report)
            with self.subTest(path=path),self.assertRaises(ValueError):add_field(report,path,'New')
            self.assertEqual(report,old)
        report={'summary':{'one_liner':'Retained'},'debates':[{'topic':'Retained'}]}
        with self.assertRaises(ValueError):add_field(report,'/summary/one_liner','Overwrite')
        add_field(report,'/debates/0/market_view','New');self.assertEqual(report['debates'][0]['topic'],'Retained')
        self.assertEqual(len(missing_fields({})),18)
