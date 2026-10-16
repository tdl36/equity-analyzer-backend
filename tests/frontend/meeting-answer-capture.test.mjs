import test from 'node:test';
import assert from 'node:assert/strict';
import {checkedAnswer,answerPayload,answerReceipt} from '../../src/meeting-answer-capture-model.mjs';
const base={id:1,meeting_id:24,question:'Synthetic question?',response_notes:null,status:'planned'};
test('exact recorded answer payload retains baseline and text',()=>{const p=answerPayload(base,' Recorded answer ');assert.deepEqual(p.expected,{responseNotes:null,status:'planned'});assert.equal(p.responseNotes,' Recorded answer ');assert.equal(p.meetingId,24);});
test('successful HTTP alone cannot confirm a different answer receipt',()=>{const p=answerPayload(base,'Saved answer');const row={...base,response_notes:'Saved answer',status:'answered'};assert.equal(answerReceipt({answer:row,replayed:false},base,p),row);for(const change of [{id:2},{meeting_id:25},{question:'Other?'},{response_notes:'Different'},{status:'planned'}])assert.throws(()=>answerReceipt({answer:{...row,...change},replayed:false},base,p));assert.throws(()=>answerReceipt({answer:row},base,p));});
test('saved identity requires a real answer record',()=>{for(const row of [null,{...base,response_notes:{}},{...base,status:null},{...base,id:'1'}])assert.throws(()=>checkedAnswer(row,24,1,base.question));assert.equal(checkedAnswer(base,24,1,base.question),base);});
