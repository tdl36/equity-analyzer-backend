/* Pie/segment rendering.
 *
 * mix_numeric is optional in practice. When Cigna's run supplied only the
 * `mix` labels, every share read as 0, the pie drew no wedges, and all three
 * labels stacked on one point -- the page printed "~17M%inimal".
 */
import { readFileSync } from 'node:fs';
import assert from 'node:assert';

const src = readFileSync(new URL('../../src/deepdive_render.js', import.meta.url), 'utf8');
// pieArc/piePoint are geometry helpers defined elsewhere in the module; stub
// them so the test exercises share parsing and label placement, not arc maths.
const pick = src.slice(src.indexOf('function hasProfitMix'), src.indexOf('function numsIn'));
const mod = new Function('esc', 'pieArc', 'piePoint',
  pick + '\nreturn {segmentShare, reportPieSVG, setSegmentBasis, hasProfitMix};')(
  s => String(s),
  () => 'M0,0',
  (cx, cy, r, deg) => {
    const rad = (deg - 90) * Math.PI / 180;
    return [(cx + r * Math.cos(rad)).toFixed(1), (cy + r * Math.sin(rad)).toFixed(1)];
  });

let failures = 0;
const test = (n, f) => { try { f(); console.log(`ok   ${n}`); }
                         catch (e) { failures++; console.log(`FAIL ${n}\n     ${e.message}`); } };

test('a share is read from the label when mix_numeric is absent', () => {
  assert.strictEqual(mod.segmentShare({ mix: '~86%' }), 86);
  assert.strictEqual(mod.segmentShare({ mix: '~17%' }), 17);
  assert.strictEqual(mod.segmentShare({ mix: 'Minimal' }), 0);
  assert.strictEqual(mod.segmentShare({ mix_numeric: 42, mix: '~86%' }), 42);
});

test('a non-numeric segment gets no label rather than one stacked on a neighbour', () => {
  const svg = mod.reportPieSVG([
    { mix: '~86%' }, { mix: '~17%' }, { mix: 'Minimal' }]);
  const labels = [...svg.matchAll(/<text[^>]*>([^<]*)</g)].map(m => m[1]);
  assert.ok(!labels.includes('Minimal'),
    `a zero-width wedge must not be labelled: got ${JSON.stringify(labels)}`);
  assert.ok(labels.includes('~86%') && labels.includes('~17%'), JSON.stringify(labels));
});

test('labels never share a position', () => {
  const svg = mod.reportPieSVG([{ mix: '~86%' }, { mix: '~17%' }, { mix: 'Minimal' }]);
  const pts = [...svg.matchAll(/<text x="([\d.]+)" y="([\d.]+)"/g)]
    .map(m => `${Math.round(+m[1])},${Math.round(+m[2])}`);
  assert.strictEqual(new Set(pts).size, pts.length, `overlapping labels at ${pts}`);
});

test('the pie shows operating profit when the company discloses it', () => {
  // Where a company earns its money is more decision-useful than where it books
  // sales, and the two diverge: Eaton's Mobility segment is 10% of revenue but
  // about 5% of segment profit.
  const etn = [
    { short_name: 'Elec Americas', mix: '47%', mix_numeric: 47, profit_mix: '55%', profit_mix_numeric: 55 },
    { short_name: 'Elec Global', mix: '29%', mix_numeric: 29, profit_mix: '26%', profit_mix_numeric: 26 },
    { short_name: 'Aerospace', mix: '14%', mix_numeric: 14, profit_mix: '14%', profit_mix_numeric: 14 },
    { short_name: 'Mobility', mix: '10%', mix_numeric: 10, profit_mix: '5%', profit_mix_numeric: 5 },
  ];
  assert.strictEqual(mod.setSegmentBasis(etn), 'profit');
  assert.strictEqual(mod.segmentShare(etn[3]), 5, 'must use the profit share');
});

test('it falls back to revenue rather than showing a partial profit split', () => {
  const partial = [
    { mix: '47%', mix_numeric: 47, profit_mix_numeric: 55 },
    { mix: '29%', mix_numeric: 29 },
    { mix: '24%', mix_numeric: 24 },
  ];
  assert.strictEqual(mod.setSegmentBasis(partial), 'revenue');
  assert.strictEqual(mod.segmentShare(partial[0]), 47);
});

test('a profit split that does not add up is rejected', () => {
  const bad = [
    { mix_numeric: 50, profit_mix_numeric: 90 },
    { mix_numeric: 50, profit_mix_numeric: 90 },
  ];
  assert.strictEqual(mod.setSegmentBasis(bad), 'revenue');
});

console.log(failures ? `\n${failures} failing` : '\nall segment tests passed');
process.exit(failures ? 1 : 0);
