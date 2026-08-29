/* Sensitivity-matrix tests.
 *
 * The matrix silently produced nonsense for CVS: EPS rows of $2018.5 and
 * $4045.5 implying $30,278 share prices for a $93 stock, because a year in
 * "$8.50 (2027E)" was parsed as an earnings figure. The DE golden fixture
 * writes "$40-45" with no year, so nothing caught it. These tests use the
 * shapes real models actually emit.
 */
import { readFileSync } from 'node:fs';
import assert from 'node:assert';

const src = readFileSync(new URL('../../src/deepdive_render.js', import.meta.url), 'utf8');
// The module targets a browser; lift out the pure helpers under test.
const pick = src.slice(src.indexOf('function finNums'), src.indexOf('function reportSensitivityWideHTML'));
const esc = s => String(s);
const parseMoneyNumber = v => { const n = (String(v||'').match(/\d+(?:\.\d+)?/g)||[]).map(Number)[0]; return Number.isFinite(n)?n:null; };
const mod = new Function('esc', 'parseMoneyNumber', pick + '\nreturn {finNums, pickSpread, reportSensitivityHTML};')(esc, parseMoneyNumber);

let failures = 0;
const test = (name, fn) => {
  try { fn(); console.log(`ok   ${name}`); }
  catch (e) { failures++; console.log(`FAIL ${name}\n     ${e.message}`); }
};

test('a year in the EPS string is not treated as earnings', () => {
  assert.deepStrictEqual(mod.finNums('$8.50 (2027E)'), [8.5]);
  assert.deepStrictEqual(mod.finNums('~89.75% (±25bps)'), [89.75]);
  assert.deepStrictEqual(mod.finNums('$40-45'), [40, 45]);
});

const cvs = {
  at_glance: { share_price: '$93.06' },
  valuation_scenarios: [
    { case: 'Bear', earnings: '$7.00 (2027E)', multiple: '9.0x' },
    { case: 'Base', earnings: '$8.50 (2027E)', multiple: '12.0x' },
    { case: 'Bull', earnings: '$9.25 (2027E)', multiple: '14.0x' },
  ],
};
const cvsFin = { historical_pe: '15–17x (pre-2024 avg.)', forward_pe: '10.9x', eps: '$7.90–$8.10' };

test('CVS rows are plausible EPS, not years', () => {
  const html = mod.reportSensitivityHTML(cvs, cvsFin);
  const rows = [...html.matchAll(/<tr><th>\$([\d.,]+)<\/th>/g)].map(m => Number(m[1]));
  assert.ok(rows.length === 4, `expected 4 rows, got ${rows.length}`);
  for (const r of rows) {
    assert.ok(r > 0 && r < 100, `EPS row ${r} is not a plausible per-share figure`);
  }
});

test('implied prices are in the same universe as the share price', () => {
  const html = mod.reportSensitivityHTML(cvs, cvsFin);
  const prices = [...html.matchAll(/<b>\$([\d,]+)<\/b>/g)].map(m => Number(m[1].replace(/,/g, '')));
  assert.ok(prices.length >= 16, `expected >=16 cells, got ${prices.length}`);
  for (const p of prices) {
    assert.ok(p > 93.06 / 10 && p < 93.06 * 10,
      `implied price $${p} is absurd against a $93.06 share price`);
  }
});

test('multiples are ascending and distinct', () => {
  const html = mod.reportSensitivityHTML(cvs, cvsFin);
  const cols = [...html.matchAll(/<th>([\d.]+)x<\/th>/g)].map(m => Number(m[1]));
  assert.strictEqual(cols.length, 4, `expected 4 multiples, got ${cols.join(',')}`);
  for (let i = 1; i < cols.length; i++) {
    assert.ok(cols[i] > cols[i - 1], `multiples not ascending: ${cols.join(', ')}`);
    assert.ok(cols[i] - cols[i - 1] >= 0.3, `multiples too clustered: ${cols.join(', ')}`);
  }
});

test('the DE golden shape still yields four sane rows', () => {
  const de = {
    at_glance: { share_price: '$400' },
    valuation_scenarios: [
      { case: 'Bear', earnings: '<$40', multiple: '15-20x' },
      { case: 'Base', earnings: '$40-45', multiple: '20-24x' },
      { case: 'Bull', earnings: '$45+', multiple: 'Premium' },
    ],
  };
  const html = mod.reportSensitivityHTML(de, { historical_pe: '', forward_pe: '', eps: '' });
  const rows = [...html.matchAll(/<tr><th>\$([\d.,]+)<\/th>/g)].map(m => Number(m[1]));
  assert.strictEqual(rows.length, 4);
  for (const r of rows) assert.ok(r > 10 && r < 200, `DE EPS row ${r} implausible`);
});

test('missing scenarios do not crash the matrix', () => {
  const html = mod.reportSensitivityHTML({ at_glance: {} }, {});
  assert.ok(html.includes('<table>'), 'should still render a table');
});

test('a one-off peak multiple does not stretch the columns', () => {
  // "18-year avg ~20x; peak 37x at Sept 2024 peak" produced a 10x/17x/22x/37x
  // header dominated by a number nobody is underwriting to.
  const html = mod.reportSensitivityHTML(
    { at_glance: { share_price: '$600' },
      valuation_scenarios: [
        { case: 'Bear', earnings: '$25 (2027E)', multiple: '14x' },
        { case: 'Base', earnings: '$30 (2027E)', multiple: '18x' },
        { case: 'Bull', earnings: '$34 (2027E)', multiple: '22x' }] },
    { historical_pe: '18-year avg ~20x; peak 37x at Sept 2024 peak',
      forward_pe: '17x', eps: '' });
  const cols = [...html.matchAll(/<th>([\d.]+)x<\/th>/g)].map(m => Number(m[1]));
  assert.ok(Math.max(...cols) <= 17 * 2.2,
    `outlier multiple survived: ${cols.join(', ')}`);
});

console.log(failures ? `\n${failures} failing` : '\nall sensitivity tests passed');
process.exit(failures ? 1 : 0);
