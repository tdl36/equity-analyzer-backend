import test from 'node:test';
import assert from 'node:assert/strict';
import {commandRecoveryKey,readCommand,retainCommand,clearCommand,commandReceipt,exactCommand} from '../../src/collection-command-recovery.mjs';
const body={requestId:'11111111-1111-4111-8111-111111111111',action:'trigger',payload:{ticker:'CVS'}};
function storage(){const data=new Map();return {getItem:k=>data.get(k)??null,setItem:(k,v)=>data.set(k,v),removeItem:k=>data.delete(k)};}
test('collection command survives reload and only exact replay can replace it',()=>{
 const s=storage(),key=commandRecoveryKey('/api');retainCommand(s,key,body);assert.deepEqual(readCommand(s,key),body);
 retainCommand(s,key,structuredClone(body));assert.throws(()=>retainCommand(s,key,{...body,payload:{ticker:'ABT'}}),/unresolved/);
 assert.throws(()=>clearCommand(s,key,{...body,action:'retry'}),/changed/);assert.deepEqual(readCommand(s,key),body);
 clearCommand(s,key,body);assert.equal(readCommand(s,key),null);assert.notEqual(commandRecoveryKey('local'),key);
});
test('corrupt or unavailable storage never silently discards recovery',()=>{
 const s=storage();for(const value of ['{','null',JSON.stringify({...body,action:'shell'}),JSON.stringify({...body,requestId:'bad'}),'x'.repeat(500001)]){s.setItem('key',value);assert.throws(()=>readCommand(s,'key'));assert.equal(s.getItem('key'),value);}
 assert.throws(()=>retainCommand({getItem:()=>null,setItem(){}},'key',body),/not be saved/);
 const noClear=storage();retainCommand(noClear,'key',body);noClear.removeItem=()=>{};assert.throws(()=>clearCommand(noClear,'key',body),/not be cleared/);
});
test('receipt recovery rejects wrong identity, ambiguous lists and unknown status',()=>{
 assert.equal(exactCommand({commands:[]},body),null);
 for(const status of ['queued','applied','failed'])assert.equal(exactCommand({commands:[{id:body.requestId,status}]},body).status,status);
 for(const row of [{id:'other',status:'queued'},{id:body.requestId,status:'complete'},null])assert.throws(()=>commandReceipt(row,body));
 for(const data of [{},null,{commands:[{id:body.requestId,status:'queued'},{id:body.requestId,status:'queued'}]}])assert.throws(()=>exactCommand(data,body));
});
