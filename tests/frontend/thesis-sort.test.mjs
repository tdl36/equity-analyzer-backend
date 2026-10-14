import test from 'node:test';
import assert from 'node:assert/strict';
import {sortTheses} from '../../src/thesis-sort.mjs';
test('thesis date order uses newest first with missing/invalid dates last and stable ticker ties', () => {
    const rows = [{ticker:'ABT',updated:'2026-01-01T00:00:00Z'}, {ticker:'UNH',updated:'2026-10-14T00:00:00Z'}, {ticker:'Z',updated:'bad'}, {ticker:'A'}, {ticker:'SYK',updated:'2026-10-13T20:00:00-04:00'}];
    assert.deepEqual(sortTheses(rows,'updated').map(r=>r.ticker), ['SYK','UNH','ABT','A','Z']);
    assert.deepEqual(sortTheses(rows,'alpha').map(r=>r.ticker), ['A','ABT','SYK','UNH','Z']);
    assert.equal(rows[0].ticker,'ABT');
});
