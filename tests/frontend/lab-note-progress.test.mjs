import test from 'node:test';import assert from 'node:assert/strict';
import {checkedNoteProgress,batchOutcome} from '../../src/lab-note-progress-model.mjs';
const data={noteId:'id',ticker:'ABT',status:'attention',readOnly:true,modelKey:'fixture',revision:2,acceptedRevision:1,batchCount:2,batches:[{batch:1,status:'complete'},{batch:2,status:'uncertain'}]};
test('exact job inspection separates saved replies, uncertainty, draft and acceptance',()=>{
 assert.deepEqual(checkedNoteProgress(data,'id','ABT'),data);
 assert.equal(batchOutcome('complete'),'Reply saved');assert.match(batchOutcome('uncertain'),/do not repeat/);
 assert.match(batchOutcome('dispatched'),/not yet confirmed/);assert.equal(batchOutcome(),'No dispatch recorded');
});
test('substituted jobs, companies and malformed batch identities fail closed',()=>{
 for(const change of [{noteId:'other'},{ticker:'UNH'},{readOnly:false},{batchCount:9},{revision:0},{acceptedRevision:'1'},{status:'complete'},
 {batches:[{batch:1,status:'complete'},{batch:1,status:'complete'}]},{batches:[{batch:3,status:'complete'}]},{batches:[{batch:1,status:'unknown'}]}])assert.throws(()=>checkedNoteProgress({...data,...change},'id','ABT'));
});
