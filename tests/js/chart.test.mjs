/* Earnings-chart annotation placement.
 *
 * Labels were drawn text-anchor="middle" at their data point with no bounds
 * check and no collision handling. The first point sits on the y-axis, so its
 * label lost the half at negative x: "Pre-Aetna headwinds" printed as
 * "re-Aetna headwinds" in shipped PDFs. Labels at the right ran over the
 * valuation panel, and neighbours overprinted each other.
 */
import { readFileSync } from 'node:fs';
import assert from 'node:assert';

const src = readFileSync(new URL('../../src/deepdive_render.js', import.meta.url), 'utf8');
const pick = src.slice(src.indexOf('function annotationsSVG'), src.indexOf('function identityRows'));
const mod = new Function('esc', pick + '\nreturn {earningsChartSVG};')(s => String(s));

const W = 430, CHAR_W = 4.6;
const boxesOf = svg => [...svg.matchAll(
  /<text x="([\d.]+)" y="([\d.]+)" text-anchor="(\w+)" class="anno">([^<]*)</g)]
  .map(([, X, Y, anchor, txt]) => {
    const w = txt.length * CHAR_W, x = Number(X);
    const x0 = anchor === 'start' ? x : anchor === 'end' ? x - w : x - w / 2;
    return { x0, x1: x0 + w, y: Number(Y), txt };
  });

const PTS = [
  { period: '2021', value: 8,    annotation: 'Pre-Aetna headwinds' },
  { period: '2022', value: 9,    annotation: 'Peak margin cycle' },
  { period: '2023', value: 6,    annotation: 'MA cost pressures emerge' },
  { period: '2024', value: 4,    annotation: 'Aetna MBR spike; restructuring' },
  { period: '2025', value: 6.75, annotation: 'Turnaround begins' },
  { period: '2026', value: 8,    kind: 'estimate', annotation: 'Guidance midpoint $7.90-$8.10' },
  { period: '2027', value: 8.44, kind: 'estimate', annotation: 'Mgmt. floor; 13% growth implied' },
  { period: '2028', value: 9.5,  kind: 'estimate', annotation: 'Mid-teens CAGR target from 2026' },
];

let failures = 0;
const test = (name, fn) => {
  try { fn(); console.log(`ok   ${name}`); }
  catch (e) { failures++; console.log(`FAIL ${name}\n     ${e.message}`); }
};

test('no annotation is drawn outside the plot', () => {
  for (const b of boxesOf(mod.earningsChartSVG({ points: PTS }))) {
    assert.ok(b.x0 >= -0.5, `"${b.txt}" starts at ${b.x0.toFixed(1)}, left of the plot`);
    assert.ok(b.x1 <= W + 0.5, `"${b.txt}" ends at ${b.x1.toFixed(1)}, right of the plot`);
  }
});

test('the first point keeps its whole label', () => {
  const svg = mod.earningsChartSVG({ points: PTS });
  assert.ok(svg.includes('Pre-Aetna headwinds'),
    'the leading annotation must not be clipped by the y-axis');
});

test('no two annotations overlap', () => {
  const b = boxesOf(mod.earningsChartSVG({ points: PTS }));
  for (let i = 0; i < b.length; i++) {
    for (let j = i + 1; j < b.length; j++) {
      const clash = !(b[i].x1 < b[j].x0 - 2 || b[i].x0 > b[j].x1 + 2)
                    && Math.abs(b[i].y - b[j].y) < 10;
      assert.ok(!clash, `"${b[i].txt}" overlaps "${b[j].txt}"`);
    }
  }
});

test('a crowded chart still places every label or drops it cleanly', () => {
  const crowded = PTS.map(p => ({ ...p, annotation: 'A fairly long annotation label here' }));
  const b = boxesOf(mod.earningsChartSVG({ points: crowded }));
  for (let i = 0; i < b.length; i++) {
    assert.ok(b[i].x0 >= -0.5 && b[i].x1 <= W + 0.5, 'dropped labels must not leak off-plot');
    for (let j = i + 1; j < b.length; j++) {
      const clash = !(b[i].x1 < b[j].x0 - 2 || b[i].x0 > b[j].x1 + 2)
                    && Math.abs(b[i].y - b[j].y) < 10;
      assert.ok(!clash, 'crowded labels must not overlap');
    }
  }
});

test('long annotations are shortened, not left to span the chart', () => {
  const long = [{ period: '2021', value: 5,
    annotation: 'An extremely long annotation that could never fit inside the plot area' },
    { period: '2022', value: 6 }];
  const b = boxesOf(mod.earningsChartSVG({ points: long }));
  assert.ok(b.length === 1 && b[0].x1 - b[0].x0 <= W, 'label must be trimmed to fit');
});

console.log(failures ? `\n${failures} failing` : '\nall chart tests passed');
process.exit(failures ? 1 : 0);
