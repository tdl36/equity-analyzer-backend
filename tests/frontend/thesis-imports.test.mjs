import test from 'node:test';
import assert from 'node:assert/strict';
import {parseDraft,canApprove} from '../../src/thesis-import-model.mjs';
import {readRoute,routeHash} from '../../src/workspace-model.mjs';

test('accepts JSON and fenced JSON but identifies wrong files and excessive uploads',()=>{
  const p={analysis:{thesis:{summary:'Test'}}};
  assert.deepEqual(parseDraft(JSON.stringify(p)),p);
  assert.deepEqual(parseDraft('```json\n'+JSON.stringify(p)+'\n```'),p);
  for(const s of ['%PDF-1.7','[]','{"sourceRegister":[]}','x'.repeat(2000001)])assert.throws(()=>parseDraft(s));
});
test('approval requires active reviewed draft, exact company and fresh baseline',()=>{
  const d={status:'pending',ticker:'TEST',stale:false};
  assert.equal(canApprove(d,true,'TEST',false),true);
  for(const args of [[null,true,'TEST',false],[d,false,'TEST',false],[d,true,'OTHER',false],[d,true,'TEST',true],[{...d,stale:true},true,'TEST',false],[{...d,status:'approved'},true,'TEST',false],[{...d,status:'dismissed'},true,'TEST',false]])assert.equal(canApprove(...args),false);
});
test('import review has a navigable company-aware route',()=>{
  assert.deepEqual(readRoute(routeHash('thesisimports','BRK.B')),{view:'thesisimports',ticker:'BRK.B'});
});

test('specific draft links accept only UUIDs and coexist with company routes', async()=>{
  const {requestedDraft}=await import('../../src/thesis-import-model.mjs');
  const id='12345678-1234-1234-1234-123456789abc';
  assert.equal(requestedDraft('?release=T124&thesisDraft='+id),id);
  assert.equal(requestedDraft(''),null);
  assert.throws(()=>requestedDraft('?thesisDraft=../../approve'));
});
