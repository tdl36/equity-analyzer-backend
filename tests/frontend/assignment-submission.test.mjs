import test from 'node:test';
import assert from 'node:assert/strict';
import {submissionKey,readSubmission,reserveSubmission,clearSubmission,matchingReceipt} from '../../src/assignment-submission.mjs';
const body={requestId:'11111111-1111-4111-8111-111111111111',ticker:'TEST',since:'2026-10-01',until:'2026-10-12',outputs:['visual'],origin:{fingerprint:'frozen'}};
function storage(){const m=new Map();return {getItem:k=>m.get(k)||null,setItem:(k,v)=>m.set(k,v),removeItem:k=>m.delete(k)};}
test('reload retains frozen request despite edited form; other companies and servers isolated',()=>{
 const s=storage(),key=submissionKey('',body.ticker);reserveSubmission(s,key,body);
 assert.deepEqual(readSubmission(s,key,'TEST'),body);
 assert.deepEqual(reserveSubmission(s,key,{...body,since:'2026-09-01',requestId:'22222222-2222-4222-8222-222222222222'}),body);
 assert.equal(readSubmission(s,submissionKey('remote','TEST'),'TEST'),null);
 assert.equal(readSubmission(s,submissionKey('','OTHER'),'OTHER'),null);
 assert.equal(matchingReceipt({id:body.requestId},body),true);
 assert.equal(matchingReceipt({id:'unrelated'},body),false);
 clearSubmission(s,key,body);assert.equal(readSubmission(s,key,'TEST'),null);
});
test('unreadable, wrong-company and unwritable storage fail closed',()=>{
 const s=storage(),key='k';s.setItem(key,'broken');assert.throws(()=>readSubmission(s,key,'TEST'));
 s.setItem(key,JSON.stringify({version:1,body}));assert.throws(()=>readSubmission(s,key,'OTHER'));
 assert.throws(()=>reserveSubmission({getItem:()=>null,setItem:()=>{throw Error('full')}},key,body));
 assert.throws(()=>clearSubmission(s,key,{...body,requestId:'other'}));assert.deepEqual(readSubmission(s,key,'TEST'),body);
});
