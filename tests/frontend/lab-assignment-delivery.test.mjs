import test from 'node:test';
import assert from 'node:assert/strict';
import {labAssignmentDelivery as delivery} from '../../src/lab-assignment-delivery.mjs';
test('Lab delivery separates saved drafts from a failed full note',()=>{
 const rows=delivery({status:'attention',input:{outputs:['visual','thesis','meeting_prep'],labFullNote:{}},result:{artifacts:[{kind:'visual',url:'/saved'}],thesisDraftId:'draft',labNoteId:'note',labNoteStatus:'attention'}});
 assert.equal(rows.find(r=>r.key==='visual').status,'Draft saved');
 assert.equal(rows.find(r=>r.key==='thesis').status,'Draft saved');
 assert.equal(rows.find(r=>r.key==='meeting_prep').status,'Not yet saved');
 assert.equal(rows.find(r=>r.key==='lab_note').status,'Needs attention');
});
test('parent completion and assigned identities alone do not prove output delivery',()=>{
 const rows=delivery({status:'complete',input:{outputs:['visual'],labFullNote:{}},result:{sources:[{}],labNoteId:'reserved',labNoteStatus:'running',artifacts:[{kind:'visual'}]}});
 assert.ok(rows.every(r=>r.status==='Receipt missing'));
 assert.deepEqual(delivery({input:{collectionOnly:true,outputs:['visual']}}),[]);
});
test('saved receipts never hide recorded output errors or imply approval',()=>{
 const rows=delivery({status:'attention',input:{outputs:['thesis']},result:{thesisDraftId:'draft',outputErrors:{thesis:'Review baseline'}}});
 assert.equal(rows[0].status,'Saved · needs attention');assert.equal(rows[0].error,'Review baseline');
 assert.equal(delivery({status:'cancelled',input:{outputs:['visual']}})[0].status,'Stopped before delivery');
});
