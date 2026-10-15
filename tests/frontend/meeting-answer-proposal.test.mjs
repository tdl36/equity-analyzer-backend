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
