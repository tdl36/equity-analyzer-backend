import test from 'node:test';import assert from 'node:assert/strict';
import {prioritizedResearch} from '../../src/research-priorities.mjs';
const now=Date.parse('2026-09-07T12:00:00Z');
test('fresh short and long exposures use absolute weights; unresearched holdings remain visible',()=>{const p=prioritizedResearch([{ticker:'DE',updated:'2026-09-01'}],[],{asOf:'2026-09-07',positions:[{ticker:'MDT',weightPct:-5}]},now);assert.equal(p.rows[0].ticker,'MDT');assert.equal(p.rows[0].score,80);assert.ok(p.rows[0].reasons.includes('5% short exposure'));});
test('stale or future holdings never boost research priority',()=>{for(const asOf of ['2026-08-01','2026-09-10']){const p=prioritizedResearch([{ticker:'MDT',updated:'2026-09-07'}],[],{asOf,positions:[{ticker:'MDT',weightPct:50}]},now);assert.equal(p.fresh,false);assert.equal(p.rows[0].score,0);}});
