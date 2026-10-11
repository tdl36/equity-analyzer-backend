import test from 'node:test';
import assert from 'node:assert/strict';
import {collectionRequests,collectionTime,collectionNext} from '../../src/collection-request-model.mjs';
test('collection scope retains attention and filters old finished work by default',()=>{
 const rows=[{id:1,ticker:'SYK',status:'complete',created:2000},{id:2,ticker:'DE',status:'attention',created:1000},{id:3,ticker:'SYK',status:'queued',created:3000},{id:4,ticker:'SYK',status:'needs_auth',created:1000}];
 assert.deepEqual(collectionRequests(rows,'syk').map(x=>x.id),[4,3]);
 assert.deepEqual(collectionRequests(rows,'SYK',true).map(x=>x.id),[4,3,1]);
 assert.equal(rows[0].id,1);assert.equal(collectionRequests(rows,'unknown').length,0);
});
test('Mac Unix seconds and server timestamps describe the same instant',()=>{
 assert.equal(collectionTime(1767225600).toISOString(),'2026-01-01T00:00:00.000Z');
 assert.equal(collectionTime('2026-01-01 00:00:00').toISOString(),'2026-01-01T00:00:00.000Z');
 assert.equal(collectionTime('bad'),null);
});
test('recovery guidance distinguishes authentication and downstream completion',()=>{
 assert.match(collectionNext({status:'needs_auth'}),/directly in AlphaSense/);
 assert.match(collectionNext({status:'attention'}),/frozen company/);
 assert.match(collectionNext({status:'complete'}),/parent assignment/);
 assert.match(collectionNext({status:'unknown'}),/no completion is inferred/);
});
