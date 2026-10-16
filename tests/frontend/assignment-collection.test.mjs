import test from 'node:test';import assert from 'node:assert/strict';
import {assignmentCollection} from '../../src/assignment-collection-model.mjs';
const now=Date.parse('2026-10-16T09:00:00Z');
const job=()=>({ticker:'CVS',input:{collectionOnly:true},result:{refreshRequestId:'11111111-1111-4111-8111-111111111111',macReportedAt:'2026-10-16T08:59:30Z',collection:{status:'queued',issue:'Another export must finish first.',sourceProgress:{documents:{staged:2,handed_off:1,duplicate:3},tasks:[{kind:'broker-report',status:'pending',expected:null}]}}}});
test('queued collection exposes actual issue and exact Operations destination without inventing receipts',()=>{
 const c=assignmentCollection(job(),now);assert.equal(c.label,'Queued for browser collection');assert.equal(c.issue,'Another export must finish first.');assert.equal(c.verified,0);assert.equal(c.fresh,true);assert.equal(c.counts.length,3);assert.equal(c.tasks[0].expected,null);
 assert.match(c.href,/ticker=CVS/);assert.match(c.href,/section=collection/);assert.match(c.href,/work=collection%3A11111111/);
});
test('old, invalid and future reports stay stale; reported complete does not imply verified originals',()=>{
 for(const stamp of ['2026-10-16T08:00:00Z','2026-10-16T09:01:00Z','bad',null]){const j=job();j.result.macReportedAt=stamp;j.result.collection.status='complete';const c=assignmentCollection(j,now);assert.equal(c.fresh,false);assert.equal(c.verified,0);assert.equal(c.label,'Browser collection reported complete');}
});
test('malformed progress is bounded by known counts and frozen-source jobs do not imply a new browser search',()=>{
 const j=job();j.result.collection.sourceProgress={documents:{staged:-1,handed_off:'2',held:1,other:99},tasks:[null,{}, {kind:'transcript',status:'pending',expected:-2}]};j.result.refreshRequestId='bad';
 const c=assignmentCollection(j,now);assert.deepEqual(c.counts,[{label:'Held originals',count:1}]);assert.equal(c.tasks.length,1);assert.equal(c.tasks[0].expected,null);assert.equal(c.href,null);
 for(const input of [{labSelection:{}},{processingProposal:{}}])assert.equal(assignmentCollection({...j,input},now),null);
 assert.equal(assignmentCollection({input:{},result:{}},now),null);
});
