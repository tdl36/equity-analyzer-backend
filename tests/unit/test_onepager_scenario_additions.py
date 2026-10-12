import copy
import unittest
import uuid
from onepager_editorial_fields import editable_fields, add_scenario_field
from onepager_corrections import preparation, prepare
from onepager_edition_contract import baseline, prepare_edition
from onepager_edition_evidence import pending_evidence
from tests.unit.test_onepager_corrections import fixture, RID
from tests.unit.test_onepager_edition_contract import AID,EID

class ScenarioAdditionTests(unittest.TestCase):
    def test_empty_pack_can_prepare_reviewable_scenario_without_mutating_report(self):
        run,doc,claim=fixture();old=copy.deepcopy(run)
        assignment={'id':AID,'ticker':'TEST','input':{'ticker':'TEST'},'result':{'reportId':RID,'visualFundamentals':{'price':100}}}
        financials={'price':100};records=[]
        for key,value in [('name','Base'),('eps_or_fcf','FY2027 adjusted EPS USD 5/share'),('valuation_multiple','20x adjusted P/E'),('implied_value','USD 100/share')]:
            body={**claim,'requestId':str(uuid.uuid4()),'path':'/scenarios/0/'+key,'statement':value}
            ident,signature,proposal=prepare(run,body,doc)
            self.assertIsNone(proposal['before'])
            records.append({'id':ident,'run_id':RID,'fingerprint':body['fingerprint'],'payload_hash':signature,'proposal':proposal})
        pack=baseline(run,assignment,financials);body={k:v for k,v in pack.items() if k!='ticker'}
        body.update(requestId=EID,correctionIds=[x['id'] for x in reversed(records)])
        docs={doc['filename']:doc}
        edition=prepare_edition(run,assignment,body,records,docs,financials)
        preview=pending_evidence(run,assignment,edition,records,docs,financials)
        self.assertEqual(preview.run['state']['report']['scenarios'][0]['implied_value'],'USD 100/share')
        self.assertEqual(len(preview.claims),4)
        self.assertTrue(all(x['reviewOrigin']=='pending_editorial' for x in preview.claims.values()))
        self.assertEqual(old,run)
        run['state']['report']['scenarios']=[{'name':'New baseline'}]
        with self.assertRaises(ValueError):prepare_edition(run,assignment,body,records,docs,financials)

    def test_additions_bounded_and_do_not_replace_structures_or_other_fields(self):
        for report in [{'scenarios':'invalid'},{'scenarios':[None]},{'scenarios':[{}, {}, {}, {}]},{'scenarios':{'bear':{}}}]:
            with self.subTest(report=report):
                self.assertNotIn('/scenarios/0/name',editable_fields(report))
                with self.assertRaises(ValueError):add_scenario_field(report,'/scenarios/0/name','Base')
        report={'scenarios':[{'name':{'invalid':'nested'},'eps_or_fcf':'Existing'}]}
        for path in ['/scenarios/0/name','/scenarios/0/eps_or_fcf','/scenarios/3/name','/scenarios/0/approval','/summary/new','/scenarios/00/name']:
            with self.subTest(path=path),self.assertRaises(ValueError):add_scenario_field(report,path,'New')

    def test_supported_containers_and_nulls_keep_indices_and_existing_values(self):
        for cases in [None,[],{},[{'name':'Bear'}],{'0':{'name':'Bear'}},[{'name':None}]]:
            report={'scenarios':copy.deepcopy(cases)}
            add_scenario_field(report,'/scenarios/2/major_risks','Synthetic risk')
            result=report['scenarios'];row=result[2] if isinstance(result,list) else result['2']
            self.assertEqual(row['major_risks'],'Synthetic risk')
            if cases and (isinstance(cases,dict) or cases[0].get('name')):
                first=result[0] if isinstance(result,list) else result['0'];self.assertEqual(first['name'],'Bear')
        run,doc,body=fixture();fields=preparation(run)['fields']
        self.assertEqual(len([f for f in fields if f['before'] is None and f['path'].startswith('/scenarios/')]),27)
        with self.assertRaises(ValueError):prepare(run,{**body,'path':'/scenarios/3/name'},doc)
