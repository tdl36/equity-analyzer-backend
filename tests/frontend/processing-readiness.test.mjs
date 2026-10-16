import test from 'node:test';import assert from 'node:assert/strict';
import {checkedProcessingReadiness} from '../../src/processing-readiness-model.mjs';
const p={id:'proposal',collectionId:'collection',ticker:'ABT',digest:'digest',revision:1};
const result={proposalId:p.id,collectionId:p.collectionId,ticker:p.ticker,digest:p.digest,revision:1,status:'pending',readOnly:true,ready:true,blockers:[],checkedAt:'2026-10-16T08:00:00Z'};
test('readiness is bound to saved identity and cannot substitute approval',()=>{
 assert.equal(checkedProcessingReadiness(result,p),result);
 for(const patch of [{proposalId:'other'},{collectionId:'other'},{ticker:'CVS'},{digest:'other'},{revision:2},{readOnly:false},{ready:false},{blockers:['Changed bytes']},{checkedAt:'bad'},{status:'dispatched'}])
  assert.throws(()=>checkedProcessingReadiness({...result,...patch},p),/does not match/);
});
test('blocked and decided results remain inspectable without enabling processing',()=>{
 for(const status of ['pending','dispatched','cancelled']){
  const blocked={...result,status,ready:false,blockers:['Review saved state']};assert.equal(checkedProcessingReadiness(blocked,p).ready,false);
 }
 assert.throws(()=>checkedProcessingReadiness({...result,ready:false,blockers:[null]},p));
});
