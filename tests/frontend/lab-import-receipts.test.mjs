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
