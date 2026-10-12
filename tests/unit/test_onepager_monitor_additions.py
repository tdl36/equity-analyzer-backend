"""Synthetic source-checked monitoring proposals, with no approval or model call."""
import copy
import unittest
from onepager_monitor_fields import missing_fields,add_field,FIELDS
from onepager_edition_evidence import pending_evidence
from tests.unit.test_onepager_edition_evidence import fixture
from research_infographic import render as illustrated
from research_onepager import render as brief


class MonitorAdditionTests(unittest.TestCase):
    def test_missing_monitor_inputs_can_be_pending_without_mutating_saved_report(self):
        for key in FIELDS:
            report={'monitor':{'kpis':[{'metric':'Old claim'}]}} if key!='metric' else {}
            args=fixture('/monitor/kpis/0/'+key,report,'Proposed monitoring '+key)
            before=copy.deepcopy(args);preview=pending_evidence(*args)
            self.assertEqual(args,before)
            self.assertEqual(preview.run['state']['report']['monitor']['kpis'][0][key],'Proposed monitoring '+key)
            self.assertEqual(preview.claims['/monitor/kpis/0/'+key]['reviewOrigin'],'pending_editorial')
        args=fixture('/monitor/kpis/0/metric',{},'Synthetic utilization metric')
        preview=pending_evidence(*args)
        for output in (brief(args[0],args[1]['input'],preview=preview),illustrated(args[0],args[1]['input'],args[5],preview=preview)):
            self.assertIn('Synthetic utilization metric',output)
            self.assertIn('Timing',output)
            self.assertIn('not accepted',output)
            self.assertIn('not established',output.lower())

    def test_invalid_containers_noncanonical_paths_and_existing_values_are_preserved(self):
        for report,path in [({'monitor':'bad'},'/monitor/kpis/0/metric'),
                            ({'monitor':{'kpis':{}}},'/monitor/kpis/0/metric'),
                            ({'monitor':{'kpis':[None]}},'/monitor/kpis/0/metric'),
                            ({'monitor':{'kpis':[{'timing':'Retained'}]}},'/monitor/kpis/0/timing'),
                            ({},'/monitor/kpis/3/metric'),({},'/monitor/kpis/00/metric'),
                            ({},'/monitor/kpis/0/approval')]:
            before=copy.deepcopy(report)
            with self.subTest(path=path),self.assertRaises(ValueError):add_field(report,path,'New')
            self.assertEqual(report,before)
        report={};add_field(report,'/monitor/kpis/2/timing','Next reported quarter')
        self.assertEqual(report['monitor']['kpis'][:2],[{},{}])
        self.assertEqual(len(missing_fields({})),18)

    def test_timing_is_rendered_only_in_its_own_anchored_row(self):
        args=fixture('/monitor/kpis/1/timing',{'monitor':{'kpis':[{},{}]}},'Unique future checkpoint')
        preview=pending_evidence(*args)
        for output in (brief(args[0],args[1]['input'],preview=preview),illustrated(args[0],args[1]['input'],args[5],preview=preview)):
            self.assertNotIn('Unique future checkpoint',output)

    def test_supported_timing_keeps_its_qualifications_in_both_renderers(self):
        args=fixture();run,assignment,_,_,_,financials=args
        values={'metric':'Utilization','current':'Q2 baseline',
                'bull_threshold':'Below baseline, conditional on enrollment mix',
                'bear_threshold':'Above baseline','timing':'Q3 results, if disclosed',
                'why_it_matters':'Tests the margin assumption'}
        run['state']['report']={'monitor':{'kpis':[values]}}
        run['state']['citations']={f'/monitor/kpis/0/{k}':dict(statement=v,review='supported',passageMatched=True,evidence=[dict(sourceId='s1',matched=True)]) for k,v in values.items()}
        for output in (brief(run,assignment['input']),illustrated(run,assignment['input'],financials)):
            self.assertIn('Q3 results, if disclosed',output)
            self.assertIn('Below baseline, conditional on enrollment mix',output)
