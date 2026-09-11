import unittest
import uuid
from datetime import datetime, timezone
from research_lifecycle import cutoff
from research_workbench import validate

class LifecycleTests(unittest.TestCase):
    def test_cutoff_requires_timezone_and_rejects_future(self):
        self.assertEqual(cutoff('2020-01-01T12:00:00Z').year,2020)
        for date in ['2020-01-01','bad','2099-01-01T00:00:00Z']:
            with self.assertRaises(ValueError):cutoff(date)
        self.assertIsNotNone(cutoff(None).tzinfo)

    def payload(self):
        return dict(requestId=str(uuid.uuid4()),id=str(uuid.uuid4()),revision=0,body=dict(kind='underweight',title='Review',owner='Me',rationale='Valuation',nextAction='Review results',status='open',dueDate='2026-09-20',caseRevision=1,assumptionId=str(uuid.uuid4()),mandate='Portfolio',benchmark='Index',asOf='2020-01-01',reason='valuation',valuationAssessment='Not compelling',holdingPct='1',benchmarkPct='3',reviewConditions=[dict(id=str(uuid.uuid4()),category='valuation',state='unassessed',trigger='Expected return exceeds hurdle')]))

    def test_condition_validation_and_decision(self):
        p=self.payload();result=validate(p)[3]
        self.assertEqual(result['activeWeightPct'],'-2')
        self.assertEqual(result['reviewConditions'][0]['state'],'unassessed')
        p['body']['reviewConditions'][0]['state']='met'
        with self.assertRaises(ValueError):validate(p)
        p['body']['reviewConditions'][0]['evidence']='My valuation model updated after reported results'
        p['body'].update(status='reviewed',outcome='Maintain')
        with self.assertRaises(ValueError):validate(p)
        p['body']['reviewDecision']='maintain';validate(p)
        p['body']['reviewConditions']*=2
        with self.assertRaises(ValueError):validate(p)

    def test_legacy_records_remain_editable(self):
        p=self.payload();p['body'].pop('reviewConditions');validate(p)
        p['body']['holdingPct']='4'
        with self.assertRaises(ValueError):validate(p)

if __name__=='__main__':unittest.main()
