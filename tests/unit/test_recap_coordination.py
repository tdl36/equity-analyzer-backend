import tempfile
import unittest
from recap_coordination import coordinate,challenges,edited,numbers

DRAFT='Revenue was $120 million in FY2026. This result guarantees durable growth with no downside risk.'
QUOTE='This result guarantees durable growth with no downside risk.'
ISSUE={'draftQuote':QUOTE,'issue':'Certainty is unsupported by this draft.','recommendation':'Retain downside uncertainty and ask for evidence.'}
FINAL='Revenue was $120 million in FY2026. This supports growth, but downside risk remains and requires monitoring.'

class CoordinationTests(unittest.TestCase):
    def call(self,prompt,tokens):
        if tokens==6000:return {'issues':[ISSUE]}
        return {'markdown':FINAL,'decisions':[{'id':'1','action':'revised','reason':'Removed unsupported certainty.'}]}
    def test_roles_preserve_lead_and_record_decisions(self):
        with tempfile.TemporaryDirectory() as root:
            final,trace,checkpoints=coordinate(DRAFT,self.call,{'sourceHash':'one'},root=root)
            self.assertEqual(final,FINAL);self.assertEqual(trace['leadDraft'],DRAFT);self.assertEqual(len(trace['roles']),3);self.assertEqual(trace['decisions'][0]['action'],'revised');self.assertEqual(len(checkpoints),2)
    def test_restart_reuses_completed_stages_without_model_calls(self):
        with tempfile.TemporaryDirectory() as root:
            coordinate(DRAFT,self.call,{'sourceHash':'one'},root=root)
            final,trace,_=coordinate(DRAFT,lambda *args:self.fail('Completed stages must not run again'),{'sourceHash':'one'},root=root)
            self.assertEqual(final,FINAL);self.assertEqual(trace['reusedStages'],['challenge','editor'])
    def test_editor_interruption_preserves_challenge(self):
        with tempfile.TemporaryDirectory() as root:
            def fail(prompt,tokens):
                if tokens!=6000:raise ValueError('interrupted')
                return {'issues':[ISSUE]}
            with self.assertRaises(ValueError):coordinate(DRAFT,fail,{'sourceHash':'one'},root=root)
            calls=[]
            def resume(prompt,tokens):calls.append(tokens);return self.call(prompt,tokens)
            coordinate(DRAFT,resume,{'sourceHash':'one'},root=root);self.assertEqual(calls,[24576])
    def test_source_identity_change_does_not_reuse_old_roles(self):
        with tempfile.TemporaryDirectory() as root:
            coordinate(DRAFT,self.call,{'sourceHash':'one'},root=root);calls=[]
            def fresh(prompt,tokens):calls.append(tokens);return self.call(prompt,tokens)
            coordinate(DRAFT,fresh,{'sourceHash':'two'},root=root);self.assertEqual(calls,[6000,24576])
    def test_unmatched_draft_quote_is_rejected(self):
        with self.assertRaises(ValueError):challenges({'issues':[{**ISSUE,'draftQuote':'A sufficiently long fabricated quotation not present in the draft.'}]},DRAFT)
    def test_editor_may_not_introduce_new_numbers(self):
        issues=challenges({'issues':[ISSUE]},DRAFT)
        with self.assertRaises(ValueError):edited({'markdown':FINAL.replace('$120','$130'),'decisions':[{'id':'1','action':'revised','reason':'Changed estimate'}]},DRAFT,issues)
        self.assertIn('15',numbers('Revenue reached 15 million.'));self.assertNotIn('1',numbers('1. First section'))
    def test_missing_decision_and_excessive_deletion_are_rejected(self):
        issues=challenges({'issues':[ISSUE]},DRAFT)
        for value in ({'markdown':FINAL,'decisions':[]},{'markdown':'Short','decisions':[{'id':'1','action':'revised','reason':'Cut'}]}):
            with self.assertRaises(ValueError):edited(value,DRAFT,issues)
    def test_no_findings_skips_unnecessary_editor_pass(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[]
            def no_findings(prompt,tokens):calls.append(tokens);return {'issues':[]}
            final,trace,_=coordinate(DRAFT,no_findings,{},root=root)
            self.assertEqual(final,DRAFT);self.assertEqual(calls,[6000]);self.assertEqual(trace['roles'][-1]['status'],'not_needed')
