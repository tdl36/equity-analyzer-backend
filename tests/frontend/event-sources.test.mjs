import test from 'node:test';import assert from 'node:assert/strict';import {eventSources} from '../../src/event-sources.mjs';
test('missing manifest is unknown, not empty iCloud',()=>assert.equal(eventSources({},'Q1').available,false));
test('event inventory isolates exact folder, includes nested sources and excludes generated recaps',()=>{
 const m={lastUpdated:'2026-09-07',files:[{folder:'Catalysts/Q1',path:'a.pdf'},{folder:'Catalysts/Q10',path:'wrong.pdf'},{folder:'Catalysts/Q1/sub',path:'sub/b.pdf'},{folder:'Catalysts/Q1',path:'RECAP_generated.pdf'}]};
 const result=eventSources(m,'Q1',['a.pdf','old.pdf']);assert.equal(result.files.length,2);assert.deepEqual(result.added,['sub/b.pdf']);assert.deepEqual(result.missing,['old.pdf']);
});
