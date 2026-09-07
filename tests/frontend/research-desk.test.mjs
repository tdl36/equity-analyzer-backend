import test from 'node:test';
import assert from 'node:assert/strict';
import {parseTickers,researchQueue,runCounts,runsNeedingStatusCheck} from '../../src/research-desk-model.mjs';
test('batch ticker parsing deduplicates while surfacing invalid symbols',()=>{
 assert.deepEqual(parseTickers('de, DE; brk.b\nCAT'),{tickers:['DE','BRK.B','CAT'],invalid:[]});
 assert.deepEqual(parseTickers('DE <bad>').invalid,['<BAD>']);
 assert.deepEqual(parseTickers('').tickers,[]);
});
test('review queue uses the production updated field and includes undated entries',()=>{
 const rows=[{ticker:'OLD',updated:'2025-12-01'},{ticker:'NEW',updated:'2026-09-01'},{ticker:'UNKNOWN'},{ticker:'FUTURE',updated:'2027-01-01'}];
 const result=researchQueue(rows,90,Date.parse('2026-09-06T12:00:00Z'));
 assert.deepEqual(result.map(r=>r.ticker),['UNKNOWN','OLD']);assert.equal(result[0].age,null);assert.ok(result[1].age>90);
 assert.equal(rows[0].age,undefined);
});
test('run statistics distinguish queued, failed and completed work',()=>{
 assert.deepEqual(runCounts(['queued','running','error','complete','other'].map(status=>({status}))),{active:2,failed:1,complete:1});
});

test('old running records are flagged without declaring them failed',()=>{
 const now=Date.parse('2026-09-06T12:00:00Z');
 const runs=[{id:1,status:'running',createdAt:'2026-09-05T12:00:00Z'},{id:2,status:'running',createdAt:'2026-09-06T11:30:00Z'},{id:3,status:'complete',createdAt:'2026-09-05T12:00:00Z'}];
 assert.deepEqual(runsNeedingStatusCheck(runs,now).map(r=>r.id),[1]);assert.equal(runs[0].status,'running');
});
