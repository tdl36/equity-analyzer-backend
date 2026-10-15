import test from 'node:test';
import assert from 'node:assert/strict';
import {emptyWorkDraft,workDraft,workRecoveryKey,validWorkRecovery,readWorkRecovery,writeWorkRecovery,clearWorkRecovery} from '../../src/research-workbench-recovery.mjs';
const id='11111111-1111-4111-8111-111111111111',requestId='22222222-2222-4222-8222-222222222222';
const value=()=>({version:1,ticker:'ABT',caseRevision:3,record:{id,revision:2},draft:{...emptyWorkDraft('followup'),title:'Draft',answerId:5,answerHash:'a'.repeat(64)},pending:null});
const storage=()=>{const data=new Map();return {getItem:k=>data.get(k)??null,setItem:(k,v)=>data.set(k,v),removeItem:k=>data.delete(k)};};
test('work recovery retains exact edit baseline and is isolated by API and company',()=>{
 const s=storage(),v=value(),key=workRecoveryKey('/api','ABT');writeWorkRecovery(s,key,v);
 assert.deepEqual(readWorkRecovery(s,key,'ABT'),v);assert.equal(readWorkRecovery(s,workRecoveryKey('/api','CVS'),'CVS'),null);
 assert.notEqual(key,workRecoveryKey('','ABT'));assert.throws(()=>readWorkRecovery(s,key,'CVS'));
 clearWorkRecovery(s,key);assert.equal(readWorkRecovery(s,key,'ABT'),null);
});
test('uncertain saves freeze request, draft, work and case revisions across reload',()=>{
 const v=value();v.pending={requestId,id,revision:2,body:{...v.draft,caseRevision:3}};
 const s=storage();writeWorkRecovery(s,'key',v);assert.deepEqual(readWorkRecovery(s,'key','ABT').pending,v.pending);
 for(const change of [r=>r.pending.id=requestId,r=>r.pending.revision++,r=>r.pending.body.caseRevision++,r=>r.pending.body.title='Changed',r=>r.pending.requestId='bad']){
  const copy=structuredClone(v);change(copy);assert.equal(validWorkRecovery(copy,'ABT'),false);
 }
});
test('draft projection excludes server evidence snapshots and derived calculations',()=>{
 const draft=workDraft({...emptyWorkDraft('model'),title:'Saved',calculation:{beforeEPS:1},answerSnapshot:{response_notes:'private'},caseRevision:2});
 assert.equal(draft.title,'Saved');assert.equal(draft.calculation,undefined);assert.equal(draft.answerSnapshot,undefined);assert.equal(draft.caseRevision,undefined);
});
test('corruption, oversized records and silent storage failures remain explicit',()=>{
 for(const raw of ['{','null','x'.repeat(1000001),JSON.stringify({...value(),pending:{}})])assert.throws(()=>readWorkRecovery({getItem:()=>raw},'key','ABT'));
 assert.throws(()=>writeWorkRecovery({setItem(){},getItem:()=>null},'key',value()),/verified/);
 assert.throws(()=>clearWorkRecovery({removeItem(){},getItem:()=>'{original}'},'key'),/cleared/);
 const v=value();v.draft.inputs={};assert.equal(validWorkRecovery(v,'ABT'),false);
});
