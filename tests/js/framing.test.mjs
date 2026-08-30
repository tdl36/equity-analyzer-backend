/* Cycle framing must follow the data, not the calibration company.
 *
 * The template was built on Deere, whose adjusted EPS swings 45 -> 16 -> 42, so
 * it hard-coded "Earnings Are Cyclical" and "Mid-Cycle EPS x P/E". On Eaton --
 * earnings up in every year of the series -- those labels assert a business
 * characteristic that is not there, and a "mid-cycle" EPS is a meaningless
 * anchor when there is no cycle.
 */
import { readFileSync } from 'node:fs';
import assert from 'node:assert';

const src = readFileSync(new URL('../../src/deepdive_render.js', import.meta.url), 'utf8');
const pick = src.slice(src.indexOf('function earningsPattern'), src.indexOf('function reportSensitivityHTML'));
const mod = new Function(pick + '\nreturn {earningsPattern, earningsLabels};')();

const series = v => ({ points: v.map(x => ({ value: x })) });
let failures = 0;
const test = (n, f) => { try { f(); console.log(`ok   ${n}`); }
                         catch (e) { failures++; console.log(`FAIL ${n}\n     ${e.message}`); } };

test('a genuinely cyclical series is called cyclical', () => {
  assert.strictEqual(
    mod.earningsPattern(series([28, 23, 36, 45, 23, 34, 16, 34.5, 19, 42.5])), 'cyclical');
});

test('a secular compounder is not', () => {
  assert.strictEqual(
    mod.earningsPattern(series([5.5, 6.2, 7.3, 8.4, 9.5, 10.8, 12.1, 13.5])), 'secular');
});

test('one dip in a rising series is not a cycle', () => {
  assert.strictEqual(
    mod.earningsPattern(series([10, 11, 12, 11.4, 13, 14, 15, 16])), 'secular');
});

test('labels follow the classification', () => {
  const cyc = mod.earningsLabels(series([28, 23, 36, 45, 23, 34, 16, 34.5, 19, 42.5]));
  assert.ok(cyc.chartTitle.includes('Cyclical'), cyc.chartTitle);
  assert.ok(cyc.matrixTitle.includes('Mid-Cycle'), cyc.matrixTitle);
  assert.ok(cyc.targetsTitle.includes('Mid-Cycle'), cyc.targetsTitle);

  const sec = mod.earningsLabels(series([5.5, 6.2, 7.3, 8.4, 9.5, 10.8, 12.1, 13.5]));
  assert.ok(!sec.chartTitle.includes('Cyclical'), sec.chartTitle);
  assert.ok(!sec.matrixTitle.includes('Mid-Cycle'), sec.matrixTitle);
  assert.ok(!sec.targetsTitle.includes('Mid-Cycle'),
    'management targets, not mid-cycle targets, for a non-cyclical name');
});

test('too short a series claims nothing', () => {
  assert.strictEqual(mod.earningsPattern(series([10, 12])), 'unknown');
  // and an unknown pattern must not assert cyclicality
  assert.ok(!mod.earningsLabels(series([10, 12])).chartTitle.includes('Cyclical'));
});

console.log(failures ? `\n${failures} failing` : '\nall framing tests passed');
process.exit(failures ? 1 : 0);
