import test from 'node:test';import assert from 'node:assert/strict';import {collectionHealth} from '../../src/collection-health.mjs';
test('connected worker does not conceal an overdue collection queue',()=>{
 const h=collectionHealth({worker:{status:'awake'},policies:[{enabled:true,hours:168,lastSuccess:null}],requests:[{status:'queued',created:0},{status:'cancelled',created:0},{status:'needs_auth',created:0}]},3600000);
 assert.equal(h.verified,0);assert.equal(h.overdue,1);assert.equal(h.pending,2);assert.equal(h.needsAttention,1);
});
