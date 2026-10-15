import test from 'node:test';
import assert from 'node:assert/strict';
import {collectionQueueMessage} from '../../src/collection-request-model.mjs';
const value=()=>({status:'queued',queueState:{asOf:100,requestStatus:'queued',state:'waiting_for_slot',defaultPosition:7,eligibleCount:20,activeCount:1,active:[{ticker:'HUM',leaseUntil:200}]}});
test('queue explains actual eligibility and lease without ETA or progress promise',()=>{
 assert.match(collectionQueueMessage(value(),110000),/position 7 of 20/);
 assert.match(collectionQueueMessage(value(),110000),/HUM/);
 assert.match(collectionQueueMessage(value(),110000),/targeted worker requests may differ/);
 assert.match(collectionQueueMessage(value(),210000),/lease has since expired/);
});
test('stale, missing, inconsistent and malformed queue details cannot imply readiness',()=>{
 assert.match(collectionQueueMessage({status:'queued'},110000),/not been reported/);
 assert.match(collectionQueueMessage(value(),400000),/stale/);
 assert.match(collectionQueueMessage({...value(),status:'attention'},110000),/inconsistent/);
 const r=value();r.queueState.defaultPosition=0;assert.match(collectionQueueMessage(r,110000),/unavailable/);
 r.queueState.defaultPosition=1;r.queueState.active=[];assert.match(collectionQueueMessage(r,110000),/unavailable/);
});
test('paused, absent policy, attention and unclaimed states remain distinct',()=>{
 for(const [state,text] of [['policy_paused',/policy is paused/],['missing_policy',/policy is missing/],['needs_attention',/excluded from automatic/],['leased',/not proof/],['eligible',/must still claim/]]){
  const r=value();r.queueState.state=state;assert.match(collectionQueueMessage(r,110000),text);
 }
});
