
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

/**
 * Put the document into print layout and hand back the undo.
 *
 * Split out of printArtifact so a headless run can render exactly what the
 * print dialog renders. Testing a copy of this logic would have tested the
 * copy; the blank-PDF bug lived in the interaction between these classes and
 * when they get removed.
 */
export const applyPrintLayout = (view) => {
    if (typeof document === 'undefined') return null;
    const viewCls = view === 'twopager' ? 'print-twopager'
                  : view === 'memo' ? 'print-report' : 'print-onepager';
    const body = document.body;
    const art = document.querySelector('.dd-artifact');
    if (!art) return null;

    // Tag the artifact's ancestor chain instead of moving it.
    //
    // The first attempt reparented the artifact to <body> so nothing could sit
    // above it. That printed a BLANK page: the artifact is React-managed, and
    // moving it with appendChild lets the reconciler remove or re-render it out
    // from under the print. Never relocate a node another library owns.
    //
    // The gap it was trying to solve is real though: hiding Charlie's chrome
    // with visibility:hidden preserves layout, so every hidden toolbar still
    // occupied space above the artifact. Marking the chain lets CSS remove the
    // chain's *siblings* from flow -- same effect, no DOM surgery, and adding a
    // class is something React tolerates.
    const chain = [];
    for (let n = art.parentElement; n && n !== body; n = n.parentElement) {
        n.classList.add('dd-print-chain');
        chain.push(n);
    }
    body.classList.add('dd-printing', viewCls);

    // Print changes the available height, so rows and fit are recomputed with
    // the print class active, exactly as the prototype did.
    try { rebalanceLongform(); } catch (e) { console.warn('rebalance before print:', e); }
    try { autoFitSections(art); } catch (e) { console.warn('autofit before print:', e); }

    let restored = false;
    const restore = () => {
        if (restored) return;
        restored = true;
        body.classList.remove('dd-printing', 'print-onepager', 'print-twopager', 'print-report');
        chain.forEach(n => n.classList.remove('dd-print-chain'));
        try { rebalanceLongform(); autoFitSections(art); } catch (e) { /* screen only */ }
    };
    return restore;
};

export const printArtifact = (view) => {
    if (typeof document === 'undefined') return;
    const restore = applyPrintLayout(view);
    if (!restore) { window.print(); return; }

    // Restore only once the print job is actually done.
    //
    // This used to also fire on a 1500ms timer. Chrome's print preview is
    // asynchronous and regenerates the PDF when the DOM changes, so on any run
    // where the user took longer than 1.5s to click Save -- which is every real
    // run -- the timer stripped .dd-printing mid-preview and the gated rules
    // above vanished. @page survived because it is not gated, which is why the
    // result was a correctly-sized BLANK page rather than an unstyled one.
    //
    // afterprint is not universal, so back it with a visibility check rather
    // than a wall-clock guess: the tab regains focus when the dialog closes.
    window.addEventListener('afterprint', restore, { once: true });
    const onFocus = () => {
        window.removeEventListener('focus', onFocus);
        setTimeout(restore, 300);
    };
    window.addEventListener('focus', onFocus);
    window.print();
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
/**
 * Report text blocks that visually collide.
 *
 * The preflight only ever asked "does the page overflow?", and answered no
 * while shipped PDFs had the sensitivity matrix printed across the valuation
 * panel and threat cards overprinting each other. Overflow and collision are
 * different failures: a fixed-height section whose content spills paints over
 * its neighbours without changing any page height, so nothing upstream
 * noticed. This measures the thing readers actually see.
 */
export const detectOverlaps = (page) => {
    if (!page || typeof document === 'undefined') return [];
    const SEL = [
        '.report-card', '.report-callout', '.report-sensitivity', '.report-val-panel',
        // The panel clips, but the blocks inside it were the ones painting over
        // the matrix -- listing only the container hid the real collision.
        '.report-valuation-summary', '.report-target-grid', '.report-matrix-title',
        '.report-cycle-note', '.report-chart-wrap', '.report-metrics', '.report-table',
        '.report-threat', '.report-catalyst', '.report-opp', '.report-pool',
        '.v21-sensitivity', '.v21-bottom-line', '.v21-final-copy', '.report-matrix-title',
        '.tp-targets', '.tp-fin-bullets', '.tp-chart-row aside', '.tp-cycle',
        '.nbv-val', '.nbv-chart', '.nbv-target',
    ].join(',');
    /* An element clipped by an ancestor still reports its full layout box, so
       comparing raw rects invents collisions between things the reader never
       sees overlapping. Intersect with every clipping ancestor to get the box
       that actually paints. */
    const visibleRect = (el) => {
        let r = el.getBoundingClientRect();
        for (let n = el.parentElement; n && n !== page.parentElement; n = n.parentElement) {
            const cs = getComputedStyle(n);
            if (/hidden|clip|auto|scroll/.test(cs.overflow + cs.overflowX + cs.overflowY)) {
                const pr = n.getBoundingClientRect();
                const left = Math.max(r.left, pr.left), top = Math.max(r.top, pr.top);
                const right = Math.min(r.right, pr.right), bottom = Math.min(r.bottom, pr.bottom);
                r = { left, top, right, bottom,
                      width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
            }
        }
        return r;
    };
    const els = Array.from(page.querySelectorAll(SEL)).filter(el => {
        const cs = getComputedStyle(el);
        if (cs.display === 'none' || cs.visibility === 'hidden') return false;
        const r = visibleRect(el);
        return r.width > 8 && r.height > 8;
    });
    const out = [];
    for (let i = 0; i < els.length; i++) {
        for (let j = i + 1; j < els.length; j++) {
            const a = els[i], b = els[j];
            // Nesting is not a collision.
            if (a.contains(b) || b.contains(a)) continue;
            const ra = visibleRect(a), rb = visibleRect(b);
            const ox = Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left);
            const oy = Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top);
            // A few px of shared edge is normal for adjacent borders.
            if (ox > 4 && oy > 4) {
                const name = el => el.className.toString().trim().split(/\s+/)[0] || el.tagName;
                out.push(`overlap: .${name(a)} and .${name(b)} share `
                    + `${Math.round(ox)}x${Math.round(oy)}px`);
            }
        }
    }
    return out;
};

export const preflightPages = (root) => {
    if (!root || typeof document === 'undefined') return [];
    const pages = Array.from(root.querySelectorAll('.op-canvas, .tp-page, .report-page'));
    if (!pages.length) return [];

    let hard = [];
    let qa = { failures: [] };
    let collisions = [];
    try { hard = strictClipFailures() || []; } catch (e) { console.warn('strictClipFailures:', e); }
    try { qa = collectLayoutQA() || { failures: [] }; } catch (e) { console.warn('collectLayoutQA:', e); }
    try { pages.forEach(pg => { collisions = collisions.concat(detectOverlaps(pg)); }); }
    catch (e) { console.warn('detectOverlaps:', e); }

    const issues = [...new Set([...(hard || []), ...((qa && qa.failures) || []), ...collisions])];
    // The prototype reports per-document, not per-page, so the findings are
    // attached to the first page rather than invented against a page each.
    return pages.map((_page, idx) => ({
        page: idx + 1,
        ok: idx === 0 ? issues.length === 0 : true,
        issues: idx === 0 ? issues : [],
        utilization: null,
    }));
};
