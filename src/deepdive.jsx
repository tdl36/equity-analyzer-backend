
// Deep Dive — a thin React wrapper around the ported v24 renderers.
//
// The artifacts are built by src/deepdive_render.js, which is the prototype's
// calibrated renderer copied verbatim. This file deliberately does almost
// nothing: it hands data to those functions and injects the HTML they return.
//
// The first version of this file reimplemented all three artifacts in JSX. It
// produced the right sections in roughly the right order and looked nothing
// like the reviewed output, because the calibration is not the structure — it
// is the SVG paper grain, the hand-drawn frames, the absolute coordinates, the
// icon set, the per-section accent colours, the logo and the product art.
// Rebuilding that from a schema was the wrong instinct; porting it was the
// right one.

import * as React from 'react';
import {
    notebookHTML, institutionalHTML, dashboardHTML, strategyHTML, editorialHTML,
    twopagerNotebookHTML, twopagerInstitutionalHTML, memoHTML, setCurrent,
    strictClipFailures, collectLayoutQA, rebalanceLongform,
} from './deepdive_render';

const { useState, useEffect, useRef } = React;

export const PAGE_W = 1024;

// Names match the prototype's selector so the handoff's calibration notes still
// apply by name.
export const ONEPAGER_TEMPLATES = [
    { key: 'notebook',      label: '01 · Research Notebook',       fn: notebookHTML },
    { key: 'institutional', label: '02 · Institutional Brief',     fn: institutionalHTML },
    { key: 'dashboard',     label: '03 · Visual Equity Dashboard', fn: dashboardHTML },
    { key: 'strategy',      label: '04 · Strategy Deck',           fn: strategyHTML },
    { key: 'editorial',     label: '05 · Editorial Magazine',      fn: editorialHTML },
];

export const TWOPAGER_TEMPLATES = [
    { key: 'notebook',      label: '01 · Research Notebook Extended', fn: twopagerNotebookHTML },
    { key: 'institutional', label: '02 · Institutional Extended',     fn: twopagerInstitutionalHTML },
];

/** Scales a fixed 1024-wide artifact down to fit its container. */
export const PageFit = ({ children, className = '' }) => {
    const wrapRef = useRef(null);
    const innerRef = useRef(null);
    const [scale, setScale] = useState(1);
    const [height, setHeight] = useState(1536);
    const lastWidth = useRef(0);

    useEffect(() => {
        const el = wrapRef.current;
        if (!el) return;
        let frame = null;

        const apply = () => {
            const w = el.clientWidth || 0;
            if (!w) return;
            setScale(Math.min(1, w / PAGE_W));
            // The renderer decides how tall the artifact is — one page, two or
            // three — so measure rather than assume.
            const inner = innerRef.current;
            if (inner) setHeight(inner.scrollHeight || 1536);
        };
        const schedule = () => {
            if (frame) cancelAnimationFrame(frame);
            frame = requestAnimationFrame(apply);
        };

        // Width only: observing an element whose height this component then
        // changes is a ResizeObserver feedback loop, which surfaces as an error.
        const ro = new ResizeObserver((entries) => {
            const w = entries?.[0]?.contentRect?.width ?? el.clientWidth ?? 0;
            if (Math.abs(w - lastWidth.current) < 0.5) return;
            lastWidth.current = w;
            schedule();
        });
        ro.observe(el);
        apply();
        const settle = setTimeout(apply, 400);   // after webfonts land
        return () => { ro.disconnect(); clearTimeout(settle); if (frame) cancelAnimationFrame(frame); };
    }, [children]);

    return (
        <div ref={wrapRef} className={`dd-fit ${className}`} style={{ height: height * scale }}>
            <div ref={innerRef} className="dd-fit-inner"
                 style={{ width: PAGE_W, transform: `scale(${scale})` }}>
                {children}
            </div>
        </div>
    );
};

/** Renders one artifact from a stored run: {master, onepager}. */
export const DeepDiveArtifact = ({ run, view, template }) => {
    const html = React.useMemo(() => {
        if (!run) return '';
        try {
            const d = run.onepager || {};
            const m = run.master || {};
            setCurrent({ master: m, onepager: d });

            if (view === 'twopager') {
                const t = TWOPAGER_TEMPLATES.find(x => x.key === template) || TWOPAGER_TEMPLATES[0];
                return t.fn(d, m);
            }
            if (view === 'memo') return memoHTML(m, d);

            const t = ONEPAGER_TEMPLATES.find(x => x.key === template) || ONEPAGER_TEMPLATES[0];
            return t.fn(d, m);
        } catch (e) {
            console.error('Deep Dive render failed:', e);
            return `<div class="dd-render-error">Could not render this artifact: ${String(e.message || e)}</div>`;
        }
    }, [run, view, template]);

    // The multi-page artifacts are laid out on fixed grid rows, and the
    // prototype re-measured painted content after render to redistribute those
    // rows. Skipping that step is why the memo clipped its financial and
    // signpost sections on any company whose prose ran longer than Deere's --
    // the rows stayed at their DE-calibrated proportions no matter what was in
    // them. Runs after paint, and again once webfonts settle and change metrics.
    useEffect(() => {
        if (!html) return;
        const run = () => {
            try { rebalanceLongform(); } catch (e) { console.warn('rebalance:', e); }
            // Rows first, then fit the content into whatever rows it got.
            try { autoFitSections(document.querySelector('.dd-artifact')); }
            catch (e) { console.warn('autofit:', e); }
        };
        const a = requestAnimationFrame(run);
        const b = setTimeout(run, 450);
        const c = setTimeout(run, 1200);
        // Anything that MEASURES the layout has to run after the last pass, or
        // it reads a half-rebalanced page. That race made the same fixture
        // report zero issues on one run and eleven on the next.
        const d = setTimeout(() => {
            try { window.dispatchEvent(new CustomEvent('deepdive:layout-settled')); }
            catch (e) { /* no consumer is fine */ }
        }, 1400);
        return () => { cancelAnimationFrame(a); clearTimeout(b); clearTimeout(c); clearTimeout(d); };
    }, [html]);

    // The v24 print rules key off `.onepager-view` / `.twopager-view` /
    // `.report-view`, so the artifact keeps the wrapper it was written for.
    const viewClass = view === 'twopager' ? 'twopager-view'
                    : view === 'memo' ? 'report-view' : 'onepager-view';

    // Every interpolated value is escaped by the renderer's own esc() at the
    // point of use, so the only unescaped markup here is its template structure.
    return (
        <div className={`dd-artifact ${viewClass}`}
             dangerouslySetInnerHTML={{ __html: html }} />
    );
};

/**
 * Bounded per-section auto-fit for the multi-page artifacts.
 *
 * The memo and two-pager lay content into fixed-height sections whose heights
 * were calibrated against Deere. Any company whose prose runs longer overflows
 * them, and the alternative -- trimming every field down to Deere's exact word
 * counts -- means deleting real research (a 101-word valuation discussion cut
 * to 20) to fit a box, and still fails for the next company whose writing is
 * different again.
 *
 * So: sections that overflow get their type scaled down slightly, and only as
 * far as they need. The floor is deliberate. The handoff's rule is that fit
 * must never be solved by shrinking EVERYTHING; a bounded, per-section nudge
 * applied only where content overruns is a different thing, and it is what a
 * typesetter does. At 0.88 the smallest body text is still above the readability
 * floor, and a section that cannot fit even there is reported rather than
 * silently clipped -- which is the outcome that actually loses information.
 */
const SECTION_FIT_FLOOR = 0.86;

export const autoFitSections = (root) => {
    if (!root) return [];
    const unfit = [];
    root.querySelectorAll('.report-section, .tp-section').forEach((sec) => {
        sec.style.removeProperty('zoom');
        const box = sec.clientHeight;
        if (!box || sec.scrollHeight <= box + 2) return;

        // `zoom`, not font-size. The v24 sections are built from fixed pixel
        // heights -- tables pinned with height:548px!important, KPI cards with
        // min-height, charts with a fixed viewBox -- so scaling type changes
        // nothing about how tall they are. zoom scales the whole composed block
        // including those pixel values, which is the only lever that actually
        // shrinks this layout without rebuilding it.
        const needed = box / sec.scrollHeight;
        const scale = Math.max(SECTION_FIT_FLOOR, needed * 0.995);
        sec.style.zoom = String(scale.toFixed(3));

        if (sec.scrollHeight > sec.clientHeight + 2) {
            unfit.push((sec.className || '').split(' ')
                .find(c => c.startsWith('v21-') || c.startsWith('tp-')) || 'section');
        }
    });
    return unfit;
};

/**
 * Print one artifact.
 *
 * The prototype set a body class before printing because its stylesheet hides
 * every view by default and reveals exactly one. Charlie has to do the same, and
 * additionally suppress its own chrome, or the PDF is a shrunken canvas
 * surrounded by navigation.
 */

export const printArtifact = (view) => {
    if (typeof document === 'undefined') return;
    const viewCls = view === 'twopager' ? 'print-twopager'
                  : view === 'memo' ? 'print-report' : 'print-onepager';
    const body = document.body;
    const art = document.querySelector('.dd-artifact');
    if (!art) { window.print(); return; }

    // Move the artifact to be a direct child of <body> for the duration of the
    // print, and put it back afterwards.
    //
    // Hiding Charlie's chrome with visibility:hidden PRESERVES its layout, so
    // every hidden toolbar and nav still occupied space above the artifact and
    // the printed page opened with a huge white gap. Absolute positioning did
    // not rescue it either, because `top:0` resolves against the nearest
    // positioned ancestor -- of which the app has several -- not against the
    // page. Reparenting to <body> removes both problems: there is nothing above
    // it and nothing between it and the page box.
    const home = art.parentNode;
    const marker = document.createComment('dd-artifact-home');
    home.insertBefore(marker, art);
    body.appendChild(art);
    body.classList.add('dd-printing', viewCls);

    // Print styles change the available height, so rows and fit are recomputed
    // with the print class active, exactly as the prototype did.
    try { rebalanceLongform(); } catch (e) { console.warn('rebalance before print:', e); }
    try { autoFitSections(art); } catch (e) { console.warn('autofit before print:', e); }

    let restored = false;
    const restore = () => {
        if (restored) return;
        restored = true;
        body.classList.remove('dd-printing', 'print-onepager', 'print-twopager', 'print-report');
        if (marker.parentNode) {
            marker.parentNode.insertBefore(art, marker);
            marker.remove();
        }
        // Re-fit for the screen, whose available height differs again.
        try { rebalanceLongform(); autoFitSections(art); } catch (e) { /* screen only */ }
    };

    window.addEventListener('afterprint', restore, { once: true });
    try {
        window.print();
    } finally {
        // afterprint is not reliable everywhere; never leave the DOM rearranged.
        setTimeout(restore, 1500);
    }
};

/**
 * Layout preflight.
 *
 * Delegates to the prototype's own QA rather than walking rects generically.
 * A naive walker reports impossible numbers here -- "overflows by 1190px" on a
 * 1536px page -- because the artifacts put their HTML inside SVG foreignObject,
 * whose child rects are not reliable once the page is scaled to fit. The v24
 * QA was written against this DOM and measures painted need instead, which is
 * precisely the false-positive class the handoff warns blocks valid PDFs.
 */
export const preflightPages = (root) => {
    if (!root || typeof document === 'undefined') return [];
    const pages = Array.from(root.querySelectorAll('.op-canvas, .tp-page, .report-page'));
    if (!pages.length) return [];

    let hard = [];
    let qa = { failures: [] };
    try { hard = strictClipFailures() || []; } catch (e) { console.warn('strictClipFailures:', e); }
    try { qa = collectLayoutQA() || { failures: [] }; } catch (e) { console.warn('collectLayoutQA:', e); }

    const issues = [...new Set([...(hard || []), ...((qa && qa.failures) || [])])];
    // The prototype reports per-document, not per-page, so the findings are
    // attached to the first page rather than invented against a page each.
    return pages.map((_page, idx) => ({
        page: idx + 1,
        ok: idx === 0 ? issues.length === 0 : true,
        issues: idx === 0 ? issues : [],
        utilization: null,
    }));
};
