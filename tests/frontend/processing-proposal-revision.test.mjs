import test from 'node:test';import assert from 'node:assert/strict';
import {proposalRevisionDraft} from '../../src/processing-proposal-revision.mjs';
import {emptyDraft,validDraft,readDraftWorkspace,writeDraftWorkspace,readSubmission} from '../../src/processing-proposal-draft.mjs';
const id='11111111-1111-4111-8111-111111111111',collection='22222222-2222-4222-8222-222222222222',hash='a'.repeat(64);
const job={id:collection,ticker:'ABT',input:{since:'2026-07-01',until:'2026-10-15'},result:{sources:[{filename:'recent.pdf',sha256:hash}]}};
const source=(id,name,sha)=>({documentId:id,reference:'charlie:thesis_'+id,filename:name,sha256:sha,bytes:12});
const proposal=()=>({id,collectionId:collection,ticker:'ABT',revision:1,digest:hash,status:'pending',sources:[source('1','recent.pdf',hash),source('2','history.pdf','b'.repeat(64))],sourceContext:{'1':{role:'collection'},'2':{role:'supplemental'}},plan:{ticker:'ABT',since:job.input.since,until:job.input.until,outputs:['summary_note','meeting_prep'],researchPurpose:{kind:'event_note',event:{title:'Guidance change',date:'2026-10-01'}},horizon:'6–9 months',instruction:'Compare guidance and prior expectations',labFullNote:{allowPdfText:true},meeting:{date:'2026-10-20',type:'expert'}}});
test('revision copies intent and exact selection roles without authorization or frozen model configuration',()=>{
 const p=proposal(),before=structuredClone(p),draft=proposalRevisionDraft(p,job);
 assert.deepEqual(draft.selected,['recent.pdf']);assert.deepEqual(draft.supplements,['2']);assert.equal(draft.purpose,'event_note');assert.equal(draft.title,'Guidance change');assert.equal(draft.horizon,'6–9 months');assert.equal(draft.fullNote,true);assert.equal(draft.allowPdfText,true);assert.equal(draft.meetingType,'expert');
 for(const key of ['requestId','digest','revision','assignmentId','sourcePreparation','modelKey'])assert.equal(draft[key],undefined);
 assert.deepEqual(p,before);draft.outputs.push('visual');assert.deepEqual(p.plan.outputs,before.plan.outputs);
 const thesis=proposal();thesis.plan.researchPurpose={kind:'thesis_review'};thesis.plan.outputs=['thesis','visual'];delete thesis.plan.meeting;thesis.plan.visualFundamentals=true;
 assert.equal(proposalRevisionDraft(thesis,job).visualFundamentals,true);assert.equal(proposalRevisionDraft(thesis,job).title,'');
});
test('legacy collected-only proposal remains usable; ambiguous role, wrong scope and unknown choices fail without guessing',()=>{
 const legacy=proposal();legacy.sources=legacy.sources.slice(0,1);delete legacy.sourceContext;assert.deepEqual(proposalRevisionDraft(legacy,job).selected,['recent.pdf']);
 for(const change of [p=>p.collectionId=id,p=>p.sourceContext['2'].role='unknown',p=>p.sourceContext['1'].role='supplemental',p=>p.sources[0].sha256='c'.repeat(64),p=>p.plan.extra='future setting',p=>p.plan.horizon=' ',p=>p.plan.horizon=null,p=>p.plan.visualFundamentals=null,p=>p.plan.labFullNote={unexpected:true}]){
  const p=proposal();change(p);assert.throws(()=>proposalRevisionDraft(p,job));
 }
});
test('older draft storage migrates new editable fields and exact pending submissions retain their bytes',()=>{
 const older=emptyDraft();delete older.horizon;delete older.visualFundamentals;
 assert.equal(validDraft({...emptyDraft(),horizon:''}).horizon,'');
 assert.equal(validDraft(older).horizon,'12–24 months');assert.equal(readDraftWorkspace({getItem:()=>JSON.stringify(older)},'key').value.visualFundamentals,false);
 let raw;const storage={getItem:()=>raw,setItem:(_,v)=>{raw=v;}};
 const event=proposalRevisionDraft(proposal(),job),thesis={...emptyDraft(),selected:['foundation.pdf']};
 writeDraftWorkspace(storage,'key',event,{thesis_review:thesis});const restored=readDraftWorkspace(storage,'key');assert.deepEqual(restored.drafts.thesis_review,thesis);assert.equal(restored.value.horizon,'6–9 months');
 const body={requestId:id,filenames:['recent.pdf'],plan:{researchPurpose:{kind:'thesis_review'},outputs:['visual'],visualFundamentals:true,horizon:'3 years'}};
 assert.deepEqual(readSubmission({getItem:()=>JSON.stringify(body)},'key').value,body);
 assert.ok(readSubmission({getItem:()=>JSON.stringify({...body,plan:{...body.plan,visualFundamentals:'yes'}})},'key').error);
});
