import test from 'node:test';
import assert from 'node:assert/strict';
import {linkedProposals,proposalLink} from '../../src/meeting-answer-proposal-model.mjs';
import {readRoute} from '../../src/workspace-model.mjs';
const id='11111111-1111-4111-8111-111111111111',work='22222222-2222-4222-8222-222222222222';
const sample=()=>({workId:work,jobs:[{id,ticker:'ABT',status:'failed',work_evidence:{id:work,revision:1}}],truncated:false,activeProposal:null});
test('linked proposal keeps exact company identity and older follow-up revision',()=>{
 const d=sample();assert.equal(linkedProposals(d,'ABT',work),d);
 assert.deepEqual(readRoute(proposalLink('ABT',id)),{view:'desk',ticker:'ABT',section:'evidence',work:'amendment:'+id});
 d.jobs=[];d.activeProposal={id,ticker:'ABT',status:'awaiting_approval'};
 assert.equal(linkedProposals(d,'ABT',work).jobs.length,0);
});
test('partial, mismatched or duplicate status cannot authorize another submission',()=>{
 for(const change of [d=>delete d.activeProposal,d=>delete d.truncated,d=>d.workId=id,d=>d.jobs[0].ticker='CVS',d=>d.jobs[0].work_evidence.id=id,d=>d.jobs[0].work_evidence.revision=0,d=>d.jobs.push(d.jobs[0]),d=>d.activeProposal={id,ticker:'CVS',status:'queued'},d=>d.jobs[0].id='bad',d=>d.jobs[0].status='unknown']){
  const d=sample();change(d);assert.throws(()=>linkedProposals(d,'ABT',work),/could not be verified/);
 }
});

import {proposalIntentKey,readProposalIntent,reserveProposalIntent,proposalReceipt,clearProposalIntent} from '../../src/meeting-answer-proposal-model.mjs';
const intent=()=>({requestId:id,workEvidence:{id:work,revision:1},instructions:''});
const memory=()=>{const m=new Map();return {getItem:k=>m.has(k)?m.get(k):null,setItem:(k,v)=>m.set(k,v),removeItem:k=>m.delete(k)};};
test('reload and changed follow-up revision retain exact original intent; scope separates API and company',()=>{
 const s=memory(),key=proposalIntentKey('/api','ABT',work),p=intent();
 assert.equal(readProposalIntent(s,key,work),null);reserveProposalIntent(s,key,work,p);
 assert.deepEqual(readProposalIntent(s,key,work),p);
 assert.deepEqual(reserveProposalIntent(s,key,work,{...p,requestId:work,workEvidence:{id:work,revision:2}}),p);
 assert.notEqual(key,proposalIntentKey('/other','ABT',work));assert.notEqual(key,proposalIntentKey('/api','CVS',work));
 assert.equal(proposalReceipt({jobs:[]},'ABT',p),null);
 assert.equal(proposalReceipt({jobs:sample().jobs},'ABT',p).id,id);
 clearProposalIntent(s,key,work,p);assert.equal(readProposalIntent(s,key,work),null);
});
test('unreadable, mismatched, unavailable and nonpersistent storage refuse unsafe submission',()=>{
 const s=memory(),key='key';
 for(const raw of ['{','null',JSON.stringify({...intent(),secret:'forbidden'}),JSON.stringify({...intent(),workEvidence:{id,revision:1}})]){
 s.setItem(key,raw);assert.throws(()=>reserveProposalIntent(s,key,work,intent()));assert.equal(s.getItem(key),raw);
 }
 assert.throws(()=>reserveProposalIntent({getItem:()=>{throw Error('denied');}},key,work,intent()));
 assert.throws(()=>reserveProposalIntent({getItem:()=>null,setItem:()=>{}},key,work,intent()));
});
test('exact receipt must match company, identity and frozen work revision before clearing',()=>{
 for(const mutate of [d=>d.jobs.push(d.jobs[0]),d=>d.jobs[0].ticker='CVS',d=>d.jobs[0].id=work,d=>d.jobs[0].work_evidence.revision=2,d=>d.jobs[0].work_evidence.id=id,d=>delete d.jobs]){
 const d={jobs:sample().jobs};mutate(d);assert.throws(()=>proposalReceipt(d,'ABT',intent()));
 }
 const s=memory();reserveProposalIntent(s,'key',work,intent());
 assert.throws(()=>clearProposalIntent(s,'key',work,{...intent(),requestId:work}));assert.deepEqual(readProposalIntent(s,'key',work),intent());
});
