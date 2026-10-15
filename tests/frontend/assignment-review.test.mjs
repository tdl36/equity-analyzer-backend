import test from 'node:test';import assert from 'node:assert/strict';
import {reviewDocuments} from '../../src/assignment-review-model.mjs';
test('recovered outputs take precedence without relinking the preserved failed original',()=>{
 const docs=reviewDocuments({id:'job',ticker:'ABT',result:{reviewPackage:{editionId:'edition',thesisDraftId:'new'},thesisDraftId:'old',artifacts:[{kind:'visual',url:'/old'}]}});
 assert.deepEqual(docs.map(d=>d.label),['One-pager','Stock summary','Thesis changes']);
 assert.match(docs[0].url,/edition\/preview\?layout=recovery$/);assert.ok(docs[0].json);assert.match(docs[2].href,/thesisDraft=new/);
});
test('partial results expose only saved deliverables and retain other requested outputs',()=>{
 assert.deepEqual(reviewDocuments({id:'job',ticker:'ABT',result:{}}),[]);
 const docs=reviewDocuments({id:'job',ticker:'ABT',result:{artifacts:[{kind:'summary_note',url:'/note'}],meetingId:'meeting'}});
 assert.deepEqual(docs.map(d=>d.id),['summary_note','meeting']);assert.equal(docs[0].url,'/note');
});

test('full-note saved receipt exposes the focused Lab route independently of parent status',()=>{
 const job={id:'job',ticker:'BRK.B',status:'attention',input:{labFullNote:{}},result:{labNoteId:'11111111-1111-4111-8111-111111111111',labNoteStatus:'draft'}};
 const docs=reviewDocuments(job);assert.deepEqual(docs.map(d=>d.id),['lab_note']);
 const route=new URLSearchParams(docs[0].href.slice(1));assert.equal(route.get('ticker'),'BRK.B');assert.equal(route.get('work'),'labnote:11111111-1111-4111-8111-111111111111');
 assert.equal(docs[0].actionLabel,'Review full Lab note');assert.match(docs[0].description,/starts no generation/);
 assert.equal(docs[0].url,undefined);
});

import {assignmentLabNote} from '../../src/assignment-review-model.mjs';
test('unconfirmed notes remain recovery states and never become saved deliverables',()=>{
 for(const status of ['queued','running','attention',undefined]){
  const job={id:'job',ticker:'ABT',status:'complete',input:{labFullNote:{}},result:{labNoteId:'11111111-1111-4111-8111-111111111111',labNoteStatus:status}};
  assert.deepEqual(reviewDocuments(job),[]);const note=assignmentLabNote(job);assert.equal(note.saved,false);assert.ok(note.href);
  assert.equal(note.status,status==='attention'?'Needs attention':'Receipt missing');
 }
 assert.equal(assignmentLabNote({input:{labFullNote:{}},result:{labNoteStatus:'draft'},status:'complete'}).status,'Receipt missing');
 assert.equal(assignmentLabNote({input:{labFullNote:{}},result:{labNoteStatus:'draft'}}).href,null);
 assert.equal(assignmentLabNote({input:{collectionOnly:true,labFullNote:{}},result:{labNoteId:'11111111-1111-4111-8111-111111111111',labNoteStatus:'draft'}}),null);
 assert.equal(assignmentLabNote({input:{labFullNote:{}},result:{labNoteId:'invalid',labNoteStatus:'draft'}}).href,null);
 assert.equal(assignmentLabNote({}),null);
});
test('missing and stopped note receipts keep partial output links',()=>{
 const job={id:'job',ticker:'ABT',status:'cancelled',input:{labFullNote:{}},result:{artifacts:[{kind:'summary_note',url:'/saved'}]}};
 assert.equal(assignmentLabNote(job).status,'Stopped before delivery');assert.equal(assignmentLabNote(job).href,null);
 assert.deepEqual(reviewDocuments(job).map(d=>d.id),['summary_note']);
});
