import test from 'node:test';
import assert from 'node:assert/strict';
import {assignmentFormKey,readAssignmentForm,writeAssignmentForm,clearAssignmentForm} from '../../src/assignment-form-draft.mjs';
const form={purpose:'event_note',eventTitle:'Guidance',eventDate:'2026-10-15',since:'2026-08-01',until:'2026-10-15',horizon:'12 months',instruction:'Check contrary evidence',meetingDate:'',meetingType:'expert',collectionOnly:true,fullNote:false,notePdfText:false,visualFundamentals:false,collectionLimit:200,chosen:['summary_note']};
const storage=()=>{const m=new Map();return {getItem:k=>m.has(k)?m.get(k):null,setItem:(k,v)=>m.set(k,v),removeItem:k=>m.delete(k)};};
test('unfinished form survives reload and isolates server, company, mode and frozen source context',()=>{
 const s=storage(),scope={api:'a',ticker:'CVS',labMode:true},key=assignmentFormKey(scope);
 writeAssignmentForm(s,key,form);assert.deepEqual(readAssignmentForm(s,key),form);
 for(const extra of [{api:'b'},{ticker:'ABT'},{labMode:false},{labSelection:{revision:1,digest:'a'}},{labSelection:{revision:2,digest:'a'}},{origin:{parentId:'old',fingerprint:'a'}},{origin:{parentId:'new',fingerprint:'a'}}])assert.equal(readAssignmentForm(s,assignmentFormKey({...scope,...extra})),null);
 assert.equal(assignmentFormKey({...scope,origin:{a:1,b:2}}),assignmentFormKey({...scope,origin:{b:2,a:1}}));
 clearAssignmentForm(s,key);assert.equal(readAssignmentForm(s,key),null);
});
test('bad recovery and authorization-bearing fields cannot be restored or overwritten silently',()=>{
 const s=storage(),key=assignmentFormKey({api:'a',ticker:'CVS'});
 for(const value of ['broken',JSON.stringify({version:2,scope:key,form}),JSON.stringify({version:1,scope:'other',form}),JSON.stringify({version:1,scope:key,form:{...form,requestId:'paid-request'}})]){s.setItem(key,value);assert.throws(()=>readAssignmentForm(s,key));assert.equal(s.getItem(key),value);}
 for(const change of [{chosen:['invalid']},{chosen:['thesis','thesis']},{purpose:'other'},{instruction:'a'.repeat(1801)},{collectionOnly:'true'},{acknowledgeRetry:true}])assert.throws(()=>writeAssignmentForm(s,key,{...form,...change}));
});
test('silent storage write and removal failures are visible',()=>{
 const s=storage(),key='k';assert.throws(()=>writeAssignmentForm({...s,setItem:()=>{}},key,form));
 writeAssignmentForm(s,key,form);assert.throws(()=>clearAssignmentForm({...s,removeItem:()=>{}},key));assert.deepEqual(readAssignmentForm(s,key),form);
});
