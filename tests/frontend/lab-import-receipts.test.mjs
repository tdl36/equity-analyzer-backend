import test from 'node:test';
import assert from 'node:assert/strict';
import {checkedLabImport,checkedLabImports} from '../../src/lab-import-receipts.mjs';
const ref='icloud:["main","a.pdf"]';
const source={reference:ref,filename:'a.pdf',sha256:'a'.repeat(64),bytes:12};
const job={requestId:'11111111-1111-4111-8111-111111111111',input:{ticker:'ABT',task:'note',revision:2},references:[ref],researchStarted:false,status:'complete',manifest:[source],receipts:[{...source,target:'charlie:thesis_1'}],completedRevision:3};
test('complete iCloud status requires matching original and destination receipts',()=>{
 assert.equal(checkedLabImport(job,'ABT'),job);
 for(const change of [{receipts:[]},{completedRevision:4},{manifest:[]},{researchStarted:true},{receipts:[{...job.receipts[0],sha256:'b'.repeat(64)}]},{receipts:[{...job.receipts[0],bytes:13}]},{receipts:[{...job.receipts[0],target:'charlie:research_1'}]}])assert.throws(()=>checkedLabImport({...job,...change},'ABT'));
 assert.throws(()=>checkedLabImport(job,'MSFT'));
 assert.throws(()=>checkedLabImport(job,'ABT',{requestId:job.requestId,task:'thesis',revision:2}));
});
test('malformed import snapshots cannot replace known progress or trigger completion',()=>{
 for(const imports of [[null],[job,job],[{...job,input:null}],[{...job,manifest:[{...source,bytes:15000001}]}]])assert.throws(()=>checkedLabImports({imports},'ABT'));
 assert.deepEqual(checkedLabImports({imports:[]},'ABT'),[]);
 assert.equal(checkedLabImport({...job,status:'queued',manifest:undefined,receipts:undefined,completedRevision:undefined},'ABT').status,'queued');
});

test('pending import reconciliation uses a read and preserves ambiguous outcomes',async()=>{
 const {reconcileLabImport}=await import('../../src/lab-import-receipts.mjs');
 const body={requestId:job.requestId,task:'note',revision:2},calls=[];
 const read=async(...args)=>{calls.push(args);return job;};
 assert.equal(await reconcileLabImport(read,'/backend','ABT',body),job);
 assert.deepEqual(calls,[['/backend',`/api/pipeline-lab/icloud-import/${job.requestId}`]]);
 assert.equal(await reconcileLabImport(async()=>{throw Object.assign(Error('missing'),{status:404});},'','ABT',body),null);
 for(const status of [401,409,500])await assert.rejects(()=>reconcileLabImport(async()=>{throw Object.assign(Error('unresolved'),{status});},'','ABT',body));
 await assert.rejects(()=>reconcileLabImport(read,'','MSFT',body));
 await assert.rejects(()=>reconcileLabImport(read,'','ABT',{...body,requestId:'../bad'}));
 assert.equal(calls.length,2);
});

test('Research import completion binds task, revision, source identities and bounded receipts',async()=>{
 const {checkedResearchImport}=await import('../../src/lab-import-receipts.mjs');
 const request={requestId:job.requestId,task:'note',revision:2};
 const row={source:'charlie:research_abc-123',target:'charlie:thesis_8',sha256:'a'.repeat(64),bytes:12};
 const receipt={requestId:request.requestId,input:{ticker:'ABT',task:'note',revision:2},revision:3,researchStarted:false,receipts:[row]};
 assert.equal(checkedResearchImport(receipt,'ABT',request,[row.source,'charlie:thesis_2']),receipt);
 for(const change of [{input:{...receipt.input,ticker:'MSFT'}},{input:{...receipt.input,task:'thesis'}},{revision:4},{researchStarted:true},{receipts:[]},{receipts:[row,row]},{receipts:[null]},{receipts:[{...row,sha256:'bad'}]},{receipts:[{...row,bytes:15000001}]},{receipts:[{...row,target:'charlie:research_8'}]}])assert.throws(()=>checkedResearchImport({...receipt,...change},'ABT',request));
 assert.throws(()=>checkedResearchImport(receipt,'ABT',request,['charlie:research_other']));
 assert.equal(checkedResearchImport({...receipt,replayed:true},'ABT',request).replayed,true);
});
