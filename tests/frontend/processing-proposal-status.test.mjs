import test from 'node:test';
import assert from 'node:assert/strict';
import {processingProposal,processingProposalList} from '../../src/processing-proposal-status.mjs';
const id='11111111-1111-4111-8111-111111111111',collection='22222222-2222-4222-8222-222222222222';
const job={id:collection,ticker:'ABT',input:{since:'2026-01-01',until:'2026-10-15'}};
const proposal=()=>({id,ticker:'ABT',collectionId:collection,status:'pending',revision:1,digest:'a'.repeat(64),plan:{ticker:job.ticker,...job.input,researchPurpose:{kind:'thesis_review'},outputs:['thesis']},sources:[{documentId:'1',reference:'charlie:thesis_1',filename:'original.pdf',sha256:'b'.repeat(64),bytes:23}],assignmentId:null});
test('processing status binds exact collection/company and supports empty lists and saved outputs',()=>{
 const p=proposal();assert.equal(processingProposal(p,job),p);assert.deepEqual(processingProposalList({proposals:[]},job),[]);
 p.status='dispatched';p.assignmentId=id;assert.equal(processingProposalList({proposals:[p]},job)[0].assignmentId,id);
 p.status='cancelled';p.assignmentId=null;assert.equal(processingProposal(p,job),p);
});
test('mismatched scope, malformed receipts and unsafe event outputs cannot authorize processing',()=>{
 for(const change of [p=>p.ticker='CVS',p=>p.collectionId=id,p=>p.id='bad',p=>p.revision=2,p=>p.digest='bad',p=>p.status='unknown',p=>p.plan.ticker='CVS',p=>p.plan.since='2025-01-01',p=>p.plan.outputs=['unknown'],p=>p.plan.outputs=['thesis','thesis'],p=>p.sources=[],p=>p.sources.push(p.sources[0]),p=>delete p.sources[0].sha256,p=>p.sources[0].documentId='bad',p=>p.sources[0].reference='charlie:thesis_2',p=>p.status='dispatched',p=>p.assignmentId='bad',p=>p.fullNote={modelKey:'model'},p=>p.plan.researchPurpose={kind:'event_note',event:{title:'Earnings',date:'2026-10-01'}}]){
 const p=proposal();change(p);assert.throws(()=>processingProposal(p,job),/could not be verified/);
 }
});
test('partial, duplicate and oversized lists are not empty or actionable',()=>{
 for(const data of [{},{proposals:null},{proposals:[proposal(),proposal()]},{proposals:Array(51).fill(proposal())}])assert.throws(()=>processingProposalList(data,job),/incomplete/);
 const p=proposal();p.plan.outputs=['summary_note'];p.plan.researchPurpose={kind:'event_note',event:{title:'Earnings',date:'2026-10-01'}};assert.equal(processingProposal(p,job),p);
});

test('history pages retain exact cursors and reject ambiguous recovery receipts',async()=>{
 const {processingProposalPage}=await import('../../src/processing-proposal-status.mjs');
 const rows=Array.from({length:50},(_,i)=>({...proposal(),id:`11111111-1111-4111-8111-${String(i).padStart(12,'0')}`}));
 const cursor=rows.at(-1).id;
 assert.equal(processingProposalPage({proposals:rows,nextCursor:cursor},job).nextCursor,cursor);
 assert.equal(processingProposalPage({proposals:[proposal()]},job).nextCursor,null);
 assert.deepEqual(processingProposalPage({proposals:[],nextCursor:null},job,{requestId:id}).rows,[]);
 for(const [data,scope] of [[{proposals:rows,nextCursor:cursor},{after:cursor}],[{proposals:rows,nextCursor:rows[0].id},{}],[{proposals:[proposal()],nextCursor:id},{}],[{proposals:rows,nextCursor:cursor},{requestId:id}],[{proposals:[proposal()]},{requestId:collection}]])assert.throws(()=>processingProposalPage(data,job,scope),/could not be verified/);
});
test('paged recovery merges duplicate identities while retaining updated server decisions',async()=>{
 const {mergeProcessingProposals}=await import('../../src/processing-proposal-status.mjs');
 const p=proposal(),other={...p,id:collection},updated={...p,status:'cancelled'};
 const rows=mergeProcessingProposals([p,other],[updated]);
 assert.equal(rows.length,2);assert.equal(rows[0].status,'cancelled');assert.equal(rows[1],other);
});
