// Deterministic golden-render harness.
//   ?view=onepager|twopager|memo  ?fixture=de|unh  ?template=...  ?chrome=1
//
// ?chrome=1 wraps the artifact in decoy nesting that mimics Charlie's real
// layout -- scrolling containers, toolbars, a fit wrapper. Without it the print
// path tests nothing useful, because the bug being chased is precisely what the
// app's chrome does to the printed page.
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { DeepDiveArtifact, PageFit, preflightPages, printArtifact, applyPrintLayout } from '../src/deepdive';
import de from '../fixtures/deepdive_de_golden.json';
import unh from '../fixtures/deepdive_unh_sample.json';
import stress from '../fixtures/deepdive_stress.json';
import ci from '../fixtures/deepdive_ci_sample.json';
import maxfx from '../fixtures/deepdive_max.json';

const q = new URLSearchParams(location.search);
const src = q.get('fixture') === 'unh' ? unh
    : q.get('fixture') === 'stress' ? stress
    : q.get('fixture') === 'ci' ? ci
    : q.get('fixture') === 'max' ? maxfx : de;
const root = document.getElementById('root');
const run = { master: src.master, onepager: src.onepager };
const view = q.get('view') || 'onepager';

const artifact = <DeepDiveArtifact run={run} view={view} template={q.get('template') || 'notebook'} />;

const tree = q.get('chrome') === '1'
    ? (
        <div style={{ height: '100vh', overflowY: 'auto', background: '#14110c' }}>
            <div style={{ padding: 24 }}>
                <div style={{ height: 60, background: '#222' }}>decoy nav</div>
                <div style={{ height: 90, background: '#333' }}>decoy toolbar</div>
                <div className="dd-stage">
                    <PageFit>{artifact}</PageFit>
                </div>
            </div>
        </div>
      )
    : artifact;

ReactDOM.render(tree, root, () => {
    const measure = () => {
        window.__ddQA = preflightPages(root); window.__ddReady = true;
        if (q.get('qa') === '1') {
            let pre = document.getElementById('__qa');
            if (!pre) { pre = document.createElement('pre'); pre.id = '__qa'; document.documentElement.appendChild(pre); }
            pre.textContent = JSON.stringify(window.__ddQA, null, 1);
        }
    };
    window.addEventListener('deepdive:layout-settled', measure);
    setTimeout(measure, 1800);
    // Exposed so a headless run can exercise the real print path.
    window.__ddPrint = () => printArtifact(view);
    // ?print=1 leaves the document in print layout permanently so
    // `chrome --print-to-pdf` renders what the print dialog would.
    if (q.get('print') === '1') {
        const go = () => {
            applyPrintLayout(view);
            // ?nozoom=1 strips the auto-fit zoom so it can be ruled in or out
            // as the cause of the rasterised, textless print output.
            if (q.get('fonts') === '1') {
                // legibility report: smallest painted text per section, in
                // printed points (canvas px * 0.75, since 1024px -> 768pt)
                const rows = [];
                const cum = (el) => { let z = 1, n = el;
                    while (n && n !== document.body) { z *= parseFloat(getComputedStyle(n).zoom) || 1; n = n.parentElement; }
                    return z; };
                document.querySelectorAll('.strict-fit, .nbv-root').forEach(sec => {
                    let min = 99, who = '';
                    sec.querySelectorAll('*').forEach(n => {
                        if (!n.textContent || !n.textContent.trim()) return;
                        if (n.children.length) return;
                        const fs = parseFloat(getComputedStyle(n).fontSize) || 0;
                        const eff = fs * cum(n);
                        if (eff > 0 && eff < min) { min = eff; who = n.textContent.trim().slice(0, 22); window.__baseAt = fs; }
                    });
                    if (min < 99) rows.push(`${(sec.className||'').split(' ')[0]}: eff=${(min*0.75).toFixed(1)}pt `
                        + `base=${((window.__baseAt||0)*0.75).toFixed(1)}pt zoom=${(min/(window.__baseAt||1)).toFixed(2)}  "${who}"`);
                });
                const pre = document.createElement('pre'); pre.id='__fonts';
                pre.textContent = rows.join('\n');
                document.documentElement.appendChild(pre);
            }
            if (q.get('fitreport') === '1') {
                const rows = [];
                // foreignObject fit report
                document.querySelectorAll('foreignObject').forEach((fo, i) => {
                    const h = parseFloat(fo.getAttribute('height'));
                    const c = fo.firstElementChild; if (!c || !h) return;
                    const z = parseFloat(c.style.zoom) || 1;
                    if (z < 1 || c.scrollHeight > h + 1)
                        rows.push(`fo[${i}] need=${c.scrollHeight} box=${h} zoom=${z} `
                            + `shortfall=${(c.scrollHeight * z - h).toFixed(0)}`);
                });
                document.querySelectorAll('.strict-fit').forEach(sec => {
                    const par = sec.parentElement; if (!par) return;
                    const z = parseFloat(sec.style.zoom) || 1;
                    let minf = 99;
                    sec.querySelectorAll('*').forEach(n => {
                        if (n.children.length || !n.textContent || !n.textContent.trim()) return;
                        const fs = parseFloat(getComputedStyle(n).fontSize) || 0;
                        if (fs > 0 && fs < minf) minf = fs;
                    });
                    const floor = Math.min(1, 10 / (minf === 99 ? 10 : minf));
                    // which direct child sticks out, and by how much
                    const er = sec.getBoundingClientRect();
                    let worst = '', by = 0;
                    Array.from(sec.children).forEach(n => {
                        const r = n.getBoundingClientRect();
                        if (r.height > 0 && r.bottom - er.bottom > by) {
                            by = r.bottom - er.bottom;
                            worst = (n.className || n.tagName).toString().split(' ')[0];
                        }
                    });
                    rows.push(`${(sec.className||'').split(' ')[0]}: natural=${sec.offsetHeight} `
                        + `avail=${par.clientHeight} zoom=${z.toFixed(2)} minFont=${minf.toFixed(1)}px `
                        + `floor=${floor.toFixed(2)}` + (by > 1 ? `  OVER by ${by.toFixed(0)}px via .${worst}` : ''));
                });
                const pre = document.createElement('pre'); pre.id='__fitreport';
                pre.textContent = rows.join('\n');
                document.documentElement.appendChild(pre);
            }
            if (q.get('measure') === '1') {
                const rows = [];
                const add = (label, el) => {
                    if (!el) { rows.push(`${label}: MISSING`); return; }
                    const r = el.getBoundingClientRect();
                    const cs = getComputedStyle(el);
                    rows.push(`${label}: top=${r.top.toFixed(1)} left=${r.left.toFixed(1)} `
                        + `w=${r.width.toFixed(1)} h=${r.height.toFixed(1)} `
                        + `pos=${cs.position} disp=${cs.display} transform=${cs.transform} `
                        + `mt=${cs.marginTop} pt=${cs.paddingTop} zoom=${cs.zoom}`);
                };
                const svg = document.querySelector('.nbv-root');
                if (svg) {
                    rows.push(`svg viewBox=${svg.getAttribute('viewBox')} `
                        + `width=${svg.getAttribute('width')} height=${svg.getAttribute('height')}`);
                    try {
                        const bb = svg.getBBox();
                        rows.push(`svg getBBox: x=${bb.x.toFixed(1)} y=${bb.y.toFixed(1)} `
                            + `w=${bb.width.toFixed(1)} h=${bb.height.toFixed(1)}`);
                    } catch (e) { rows.push('svg getBBox failed: ' + e.message); }
                    Array.from(svg.children).forEach((c, i) => {
                        let bb = null;
                        try { bb = c.getBBox(); } catch (e) { /* defs */ }
                        const r = c.getBoundingClientRect();
                        rows.push(`  svg>${c.tagName}[${i}] cls=${c.getAttribute('class') || '-'} `
                            + (bb ? `bbox(y=${bb.y.toFixed(1)},h=${bb.height.toFixed(1)}) ` : 'bbox(n/a) ')
                            + `rect(top=${r.top.toFixed(1)},h=${r.height.toFixed(1)})`);
                    });
                }
                Array.from(document.body.children).forEach((c, i) =>
                    add(`body>child[${i}] .${c.className || '(none)'}`, c));
                ['.dd-stage', '.dd-fit', '.dd-fit-inner', '.dd-artifact',
                 '.op-canvas', '.nbv-root',
                 '.v21-fin-core', '.report-chart-wrap', '.report-cycle-note',
                 '.report-val-panel', '.report-valuation-summary',
                 '.v21-sensitivity', '.report-sensitivity',
                 '.report-sensitivity table',
                 '.t16-bottom', '.tp-final', '.tp-cases', '.tp-threats',
                 '.tp-cases > div'].forEach(sel => add(sel, document.querySelector(sel)));
                const pre = document.createElement('pre');
                pre.id = '__measure';
                pre.textContent = rows.join('\n');
                document.documentElement.appendChild(pre);
            }
            if (q.get('nozoom') === '1') {
                document.querySelectorAll('[style*="zoom"]').forEach(
                    el => el.style.removeProperty('zoom'));
            }
        };
        window.addEventListener('deepdive:layout-settled', go);
        setTimeout(go, 2000);
    }
});
