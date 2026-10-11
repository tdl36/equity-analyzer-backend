import test from 'node:test';
import assert from 'node:assert/strict';
import {recoveryEvidence} from '../../src/research-recovery-model.mjs';
test('legacy unknown return requires reconciliation, never claims safe retry',()=>{
 const e=recoveryEvidence({state:{inFlight:'report-0'}});
 assert.match(e.title,/unresolved/);assert.match(e.next,/provider usage/);assert.equal(e.hasReturn,false);
});
test('saved interrupted response differs from known failed response',()=>{
 const responseRecovery={savedStages:['report-0'],receipts:[],previousAttempts:1};
 const interrupted=recoveryEvidence({state:{inFlight:'report-0'},responseRecovery});
 assert.match(interrupted.next,/reuse/);assert.match(interrupted.next,/later stages may use/);
 const failed=recoveryEvidence({state:{knownFailure:{stage:'report-0'}},responseRecovery});
 assert.match(failed.next,/new paid attempt/);assert.equal(failed.previousAttempts,1);
});
test('a different saved stage does not resolve the interrupted call',()=>{
 const e=recoveryEvidence({state:{inFlight:'review-0'},responseRecovery:{savedStages:['report-0']}});
 assert.equal(e.hasReturn,false);assert.match(e.title,/unresolved/);
});
