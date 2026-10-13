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
