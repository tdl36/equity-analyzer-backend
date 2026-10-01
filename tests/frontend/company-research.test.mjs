import test from 'node:test';import assert from 'node:assert/strict';
import {researchProgress,researchDocument,researchCaseDraft} from '../../src/company-research-model.mjs';
const run={id:'revision-a',ticker:'ABC',created_at:'2026-10-01',status:'complete',baseline:{revision:0},sources:[{id:'s',filename:'Original',originalHash:'hash'}],state:{sections:[{id:'summary',title:'PM summary',gaps:['Missing consensus'],claims:[{id:'a',statement:'Reviewed claim',basis:'interpretation',review:'supported',passageMatched:true,evidence:[]},{id:'b',statement:'UNSUPPORTED <script>bad</script>',basis:'reported_fact',review:'needs_review',passageMatched:true,evidence:[]}]}]}};
test('compact map excludes unresolved claims; full report labels and escapes them',()=>{
 assert.equal(researchProgress(run).unresolved,1);
 assert.ok(!researchDocument(run,true).includes('UNSUPPORTED'));
 assert.ok(researchDocument(run,false).includes('NEEDS REVIEW'));
 assert.ok(researchDocument(run,false).includes('&lt;script&gt;'));
 assert.ok(researchDocument(run,true).includes('revision-a'));
 assert.ok(researchDocument(run,true).includes('Missing consensus'));
});
test('initial case draft includes only supported claims and never accepts research automatically',()=>{
 const draft=researchCaseDraft(run,()=> 'uuid');assert.equal(draft.thesis,'Reviewed claim');assert.equal(draft.assumptions.length,1);
 assert.equal(draft.assumptions[0].evidenceType,'interpretation');assert.ok(!draft.evidenceLinks);
 assert.throws(()=>researchCaseDraft({...run,baseline:{revision:1}},()=> 'uuid'));
 assert.throws(()=>researchCaseDraft({...run,status:'running'},()=> 'uuid'));
});
