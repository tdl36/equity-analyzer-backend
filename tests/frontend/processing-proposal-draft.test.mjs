import test from 'node:test';import assert from 'node:assert/strict';
import {emptyDraft,draftKey,validDraft,readDraft,writeDraft} from '../../src/processing-proposal-draft.mjs';
const storage=()=>{const data=new Map();return {getItem:k=>data.get(k),setItem:(k,v)=>data.set(k,v),removeItem:k=>data.delete(k)}};
test('authoring choices survive reload and remain collection/API scoped',()=>{
 const store=storage(),key=draftKey('local','one'),d={...emptyDraft(),selected:['original.pdf'],supplements:['12'],fullNote:true};
 writeDraft(store,key,d);assert.deepEqual(readDraft(store,key).value,d);assert.equal(readDraft(store,draftKey('prod','one')).value,null);assert.equal(readDraft(store,draftKey('local','two')).value,null);
 writeDraft(store,key,emptyDraft());assert.equal(readDraft(store,key).value,null);
});
test('restored authoring never contains request identity, approval, hashes or source bodies',()=>{
 for(const extra of [{requestId:'uuid'},{approved:true},{digest:'a'.repeat(64)},{file_data:'private'}])assert.equal(validDraft({...emptyDraft(),...extra}),null);
 const d={...emptyDraft(),selected:['one']},clean=validDraft(d);clean.selected.push('two');assert.deepEqual(d.selected,['one']);
});
test('malformed, oversized, cross-purpose and quota-failed drafts stay explicit',()=>{
 for(const change of [{selected:['one','one']},{supplements:['../12']},{fullNote:'true'},{instruction:'x'.repeat(1801)},{purpose:'event_note',outputs:['thesis']},{version:2}])assert.equal(validDraft({...emptyDraft(),...change}),null);
 const store=storage();store.setItem('bad','{bad');assert.ok(readDraft(store,'bad').error);
 const denied={getItem(){throw Error('denied')},setItem(){throw Error('quota')},removeItem(){throw Error('denied')}};
 assert.ok(readDraft(denied,'bad').error);assert.throws(()=>writeDraft(denied,'key',{...emptyDraft(),selected:['one']}),/quota/);
});

test('submission recovery retains exact replay and refuses damaged or unavailable storage',async()=>{
 const {readSubmission}=await import('../../src/processing-proposal-draft.mjs');
 const body={requestId:'11111111-1111-4111-8111-111111111111',filenames:['one.pdf'],plan:{researchPurpose:{kind:'thesis_review'},outputs:['thesis']}};
 const storage=raw=>({getItem:()=>raw});
 assert.deepEqual(readSubmission(storage(JSON.stringify(body)),'key'),{value:body,error:''});
 const note={...body,supplementalIds:['2'],plan:{researchPurpose:{kind:'event_note',event:{title:'Earnings',date:'2026-10-01'}},outputs:['summary_note','meeting_prep'],labFullNote:{allowPdfText:true},meeting:{date:'2026-10-20',type:'expert'},instruction:'Context',horizon:'12–24 months'}};
 assert.deepEqual(readSubmission(storage(JSON.stringify(note)),'key').value,note);
 for(const raw of ['{','null','[]',JSON.stringify({...body,requestId:'bad'}),JSON.stringify({...body,filenames:[]}),JSON.stringify({...body,plan:{...body.plan,outputs:['trade']}}),'x'.repeat(3000001)])assert.ok(readSubmission(storage(raw),'key').error);
 assert.ok(readSubmission({getItem(){throw Error('denied');}},'key').error);
 assert.deepEqual(readSubmission(storage(null),'key'),{value:null,error:''});
});

test('purpose workspace migrates legacy drafts without copying evidence or instructions',async()=>{
 const {readDraftWorkspace,writeDraftWorkspace,emptyPurposeDraft}=await import('../../src/processing-proposal-draft.mjs');
 let raw=JSON.stringify({...emptyDraft(),selected:['foundation.pdf'],instruction:'Longitudinal view',outputs:['thesis','visual']});
 const storage={getItem:()=>raw,setItem:(_,value)=>{raw=value;}};
 const restored=readDraftWorkspace(storage,'key');assert.equal(restored.value.purpose,'thesis_review');assert.equal(restored.drafts.event_note,undefined);
 const event={...emptyPurposeDraft('event_note'),selected:['earnings.pdf'],title:'Earnings',eventDate:'2026-10-15',instruction:'Guidance changes'};
 writeDraftWorkspace(storage,'key',event,restored.drafts);
 let next=readDraftWorkspace(storage,'key');assert.deepEqual(next.value,event);assert.deepEqual(next.drafts.thesis_review,restored.value);
 writeDraftWorkspace(storage,'key',emptyPurposeDraft('event_note'),next.drafts);
 next=readDraftWorkspace(storage,'key');assert.deepEqual(next.value.selected,[]);assert.deepEqual(next.drafts.thesis_review,restored.value);
 assert.deepEqual(emptyPurposeDraft('event_note').outputs,['summary_note']);assert.equal(emptyPurposeDraft('event_note').instruction,'');
});
test('purpose workspace rejects mixed identities and silent storage failures without repairing data',async()=>{
 const {readDraftWorkspace,writeDraftWorkspace,emptyPurposeDraft}=await import('../../src/processing-proposal-draft.mjs');
 const valid={version:2,activePurpose:'thesis_review',drafts:{thesis_review:emptyDraft()}};
 for(const change of [d=>d.drafts.thesis_review.purpose='event_note',d=>d.activePurpose='event_note',d=>d.drafts.other=emptyDraft(),d=>d.drafts.thesis_review.requestId='unexpected']){
  const d=structuredClone(valid);change(d);assert.ok(readDraftWorkspace({getItem:()=>JSON.stringify(d)},'key').error);
 }
 assert.throws(()=>writeDraftWorkspace({getItem:()=>null,setItem(){}},'key',emptyPurposeDraft('event_note'),{}),/verified/);
 assert.ok(readDraftWorkspace({getItem:()=>'{bad'},'key').error);
});
