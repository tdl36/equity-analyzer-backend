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
import { notebookHTML, institutionalHTML, dashboardHTML, strategyHTML, editorialHTML, twopagerNotebookHTML, twopagerInstitutionalHTML, memoHTML, setCurrent, strictClipFailures, collectLayoutQA, rebalanceLongform } from './deepdive_render';
var {
  useState,
  useEffect,
  useRef
} = React;
export var PAGE_W = 1024;

// Names match the prototype's selector so the handoff's calibration notes still
// apply by name.
export var ONEPAGER_TEMPLATES = [{
  key: 'notebook',
  label: '01 · Research Notebook',
  fn: notebookHTML
}, {
  key: 'institutional',
  label: '02 · Institutional Brief',
  fn: institutionalHTML
}, {
  key: 'dashboard',
  label: '03 · Visual Equity Dashboard',
  fn: dashboardHTML
}, {
  key: 'strategy',
  label: '04 · Strategy Deck',
  fn: strategyHTML
}, {
  key: 'editorial',
  label: '05 · Editorial Magazine',
  fn: editorialHTML
}];
export var TWOPAGER_TEMPLATES = [{
  key: 'notebook',
  label: '01 · Research Notebook Extended',
  fn: twopagerNotebookHTML
}, {
  key: 'institutional',
  label: '02 · Institutional Extended',
  fn: twopagerInstitutionalHTML
}];

/** Scales a fixed 1024-wide artifact down to fit its container. */
/**
 * Fit the artifact to the viewport, with a zoom the reader controls.
 *
 * The pages are 1024px wide, so on a phone they land at roughly a third size
 * and the body text becomes unreadable no matter how well it is laid out.
 * Fitting is the right default, but it cannot be the only option: a reader
 * needs to get close to a signpost table or a chart. `zoom` multiplies the fit
 * scale, and above 1 the wrapper scrolls in both axes so the page can be
 * panned rather than squeezed.
 */
export var PageFit = ({
  children,
  className = '',
  zoom = 1,
  onZoom = null
}) => {
  var wrapRef = useRef(null);
  var innerRef = useRef(null);
  var [scale, setScale] = useState(1);
  var [height, setHeight] = useState(1536);
  var lastWidth = useRef(0);
  var pinch = useRef(null);
  useEffect(() => {
    var el = wrapRef.current;
    if (!el) return;
    var frame = null;
    var apply = () => {
      var w = el.clientWidth || 0;
      if (!w) return;
      setScale(Math.min(1, w / PAGE_W));
      // The renderer decides how tall the artifact is — one page, two or
      // three — so measure rather than assume.
      var inner = innerRef.current;
      if (inner) setHeight(inner.scrollHeight || 1536);
    };
    var schedule = () => {
      if (frame) cancelAnimationFrame(frame);
      frame = requestAnimationFrame(apply);
    };

    // Width only: observing an element whose height this component then
    // changes is a ResizeObserver feedback loop, which surfaces as an error.
    var ro = new ResizeObserver(entries => {
      var w = entries?.[0]?.contentRect?.width ?? el.clientWidth ?? 0;
      if (Math.abs(w - lastWidth.current) < 0.5) return;
      lastWidth.current = w;
      schedule();
    });
    ro.observe(el);
    apply();
    var settle = setTimeout(apply, 400); // after webfonts land
    return () => {
      ro.disconnect();
      clearTimeout(settle);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [children]);

  // Pinch to zoom. Native pinch does not work on a transformed element, and
  // on a phone this is the gesture people reach for first, so track the two
  // touch points directly.
  useEffect(() => {
    var el = wrapRef.current;
    if (!el || !onZoom) return;
    var dist = t => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    var start = e => {
      if (e.touches.length === 2) pinch.current = {
        d: dist(e.touches),
        z: zoom
      };
    };
    var move = e => {
      if (e.touches.length !== 2 || !pinch.current) return;
      e.preventDefault();
      var ratio = dist(e.touches) / (pinch.current.d || 1);
      onZoom(clampZoom(pinch.current.z * ratio));
    };
    var end = () => {
      pinch.current = null;
    };
    el.addEventListener('touchstart', start, {
      passive: true
    });
    el.addEventListener('touchmove', move, {
      passive: false
    });
    el.addEventListener('touchend', end, {
      passive: true
    });
    return () => {
      el.removeEventListener('touchstart', start);
      el.removeEventListener('touchmove', move);
      el.removeEventListener('touchend', end);
    };
  }, [zoom, onZoom]);
  var effective = scale * zoom;
  return /*#__PURE__*/React.createElement("div", {
    ref: wrapRef,
    className: `dd-fit ${zoom > 1 ? 'dd-fit-zoomed' : ''} ${className}`,
    style: {
      height: height * effective
    }
  }, /*#__PURE__*/React.createElement("div", {
    ref: innerRef,
    className: "dd-fit-inner",
    style: {
      width: PAGE_W,
      transform: `scale(${effective})`
    }
  }, children));
};
export var ZOOM_MIN = 0.5;
export var ZOOM_MAX = 4;
export var clampZoom = z => Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, Math.round(z * 100) / 100));

/** Renders one artifact from a stored run: {master, onepager}. */
export var DeepDiveArtifact = ({
  run,
  view,
  template
}) => {
  var html = React.useMemo(() => {
    if (!run) return '';
    try {
      var d = run.onepager || {};
      var m = run.master || {};
      setCurrent({
        master: m,
        onepager: d
      });
      if (view === 'twopager') {
        var _t = TWOPAGER_TEMPLATES.find(x => x.key === template) || TWOPAGER_TEMPLATES[0];
        return _t.fn(d, m);
      }
      if (view === 'memo') return memoHTML(m, d);
      var t = ONEPAGER_TEMPLATES.find(x => x.key === template) || ONEPAGER_TEMPLATES[0];
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
    var run = () => {
      fitArtifact(document.querySelector('.dd-artifact'));
    };
    var a = requestAnimationFrame(run);
    var b = setTimeout(run, 450);
    var c = setTimeout(run, 1200);
    // Anything that MEASURES the layout has to run after the last pass, or
    // it reads a half-rebalanced page. That race made the same fixture
    // report zero issues on one run and eleven on the next.
    var d = setTimeout(() => {
      try {
        window.dispatchEvent(new CustomEvent('deepdive:layout-settled'));
      } catch (e) {/* no consumer is fine */}
    }, 1400);
    return () => {
      cancelAnimationFrame(a);
      clearTimeout(b);
      clearTimeout(c);
      clearTimeout(d);
    };
  }, [html]);

  // The v24 print rules key off `.onepager-view` / `.twopager-view` /
  // `.report-view`, so the artifact keeps the wrapper it was written for.
  var viewClass = view === 'twopager' ? 'twopager-view' : view === 'memo' ? 'report-view' : 'onepager-view';

  // Every interpolated value is escaped by the renderer's own esc() at the
  // point of use, so the only unescaped markup here is its template structure.
  return /*#__PURE__*/React.createElement("div", {
    className: `dd-artifact ${viewClass}`,
    dangerouslySetInnerHTML: {
      __html: html
    }
  });
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
// Lowered from 0.86. The floor is the point past which the layout stops
// shrinking and starts losing content: at 0.86 the two-pager's Final Takeaway
// could not fit five bull and five bear points, so the last of each was simply
// not printed. Type at 0.74 of nominal is still comfortably legible on a
// 1024px canvas, and losing a research point is not a trade worth making to
// keep a font one point larger.
var SECTION_FIT_FLOOR = 0.55;

/* Blocks whose height is fixed by the canvas and whose content is prose.
   These are the boxes that were clipping: a card, a table cell, a callout. */
var FIT_BLOCK_SEL = ['.report-pool', '.report-opp', '.report-catalyst', '.report-threat', '.report-callout', '.v21-other', '.v21-final-copy', '.v21-bottom-line', '.report-segments > div', '.report-signposts td', '.report-decision-lens li', '.tp-pool', '.tp-opp', '.tp-threat', '.tp-final', '.t16-note', '.tp-segments > div', '.tp-signposts td', '.nbv-opp', '.nbv-seg'].join(',');
var BLOCK_FIT_FLOOR = 0.55;
// The one-pager is the densest canvas, so it is allowed to scale further
// before anything is cut.
var FOREIGN_FIT_FLOOR = 0.50;

/**
 * Shrink a block's type until its text fits, instead of deleting the text.
 *
 * The canvas is fixed, so a box that is too small for its prose has only three
 * possible outcomes: clip the words, cut the words upstream, or make the words
 * smaller. The first two are what shipped -- reports reading "...(rebates,
 * discounts), designs..." and "Specialty biosimilar..." -- and both destroy
 * research content to protect a rectangle. Scaling the type keeps every word
 * and costs a point or two of size on the handful of blocks that need it.
 *
 * `zoom` rather than font-size because the v24 blocks are built from fixed
 * pixel values that type scaling does not move.
 */
export var autoFitBlocks = root => {
  if (!root || typeof document === 'undefined') return [];
  var unfit = [];
  root.querySelectorAll(FIT_BLOCK_SEL).forEach(el => {
    // Sections belong to autoFitSections/autoFitAgainstParent. This pass
    // used to clear their zoom and then return early -- a height:100%
    // section reports scrollHeight === clientHeight -- which wiped the fit
    // the section pass had just applied and put the content back over the
    // page edge.
    if (el.classList.contains('strict-fit')) return;
    el.style.removeProperty('zoom');
    var box = el.clientHeight;
    if (!box) return;
    // A block only needs fitting if its content is taller than its box.
    if (el.scrollHeight <= box + 1) return;
    var bfloor = floorFor(el, BLOCK_FIT_FLOOR);
    var lo = bfloor,
      hi = 1,
      best = bfloor;
    for (var i = 0; i < 7; i++) {
      var k = (lo + hi) / 2;
      el.style.zoom = String(k.toFixed(3));
      if (el.scrollHeight <= el.clientHeight + 1) {
        best = k;
        lo = k;
      } else {
        hi = k;
      }
    }
    el.style.zoom = String(best.toFixed(3));
    if (el.scrollHeight > el.clientHeight + 1) {
      unfit.push((el.className || '').toString().split(' ')[0] || el.tagName);
    }
  });
  return unfit;
};

/**
 * Fit the one-pager's boxes, which are SVG foreignObjects.
 *
 * The notebook one-pager places HTML inside foreignObject elements at absolute
 * coordinates with a fixed width and height. A foreignObject does not clip its
 * content and its child div is unconstrained, so an overfull box reports
 * scrollHeight === clientHeight -- it looks like it fits while its text paints
 * straight over the box below. That is why the only ways to keep this page
 * readable were to cut the prose upstream or let it collide.
 *
 * Constraining the child to the declared height makes the overflow measurable,
 * and then the type can be scaled to hold the words instead of deleting them.
 */
export var autoFitForeignObjects = root => {
  if (!root || typeof document === 'undefined') return [];
  var unfit = [];
  root.querySelectorAll('foreignObject').forEach(fo => {
    var h = parseFloat(fo.getAttribute('height'));
    var child = fo.firstElementChild;
    if (!h || !child) return;
    child.style.removeProperty('zoom');
    child.style.maxHeight = `${h}px`;
    child.style.overflow = 'hidden';
    if (child.scrollHeight <= h + 1) return;

    // zoom multiplies painted size, so content fits when scrollHeight * k
    // is inside the box. scrollHeight itself stays in unzoomed units.
    var ffloor = floorFor(child, FOREIGN_FIT_FLOOR);
    var lo = ffloor,
      hi = 1,
      best = ffloor;
    for (var i = 0; i < 7; i++) {
      var k = (lo + hi) / 2;
      child.style.zoom = String(k.toFixed(3));
      if (child.scrollHeight * k <= h + 1) {
        best = k;
        lo = k;
      } else {
        hi = k;
      }
    }
    child.style.zoom = String(best.toFixed(3));
    // max-height is resolved in the child's OWN coordinate space, which
    // zoom has just scaled. Leaving it at h clipped the box at h * zoom of
    // real height -- the one-pager's bull case lost its last point and cut
    // the fourth mid-word while the fit check reported success.
    child.style.maxHeight = `${(h / best).toFixed(1)}px`;
    if (child.scrollHeight * best > h + 1) {
      unfit.push(child.className || 'foreignObject');
    }
  });
  return unfit;
};
var PAGE_FIT_FLOOR = 0.62;
var PAGE_HEIGHT_PX = 1536;

/**
 * Keep a page's content on its page.
 *
 * The pages are declared with min-height, not height, so an over-full page
 * simply grows: scrollHeight equals clientHeight and every overflow check says
 * it fits, while everything past 1536px falls outside the printed sheet and is
 * silently dropped. That is how the two-pager lost the last bull and bear point
 * and the fourth threat's body while reporting a clean layout.
 *
 * Measuring the content against the real page height and scaling the page's
 * main block is the only check that sees this, and scaling beats dropping:
 * a point or two of type costs the reader far less than a missing kill
 * criterion. The header and footer are left alone so pages stay aligned.
 */
export var autoFitPages = root => {
  if (!root || typeof document === 'undefined') return [];
  var unfit = [];
  root.querySelectorAll('.tp-page, .report-page').forEach(pg => {
    var main = pg.querySelector('main') || pg;
    main.style.removeProperty('zoom');
    var chrome = pg.scrollHeight - main.scrollHeight; // header + footer
    var available = PAGE_HEIGHT_PX - Math.max(0, chrome);
    var needed = main.scrollHeight;
    if (!needed || needed <= available + 1) return;
    var k = Math.max(floorFor(main, PAGE_FIT_FLOOR), (available - 2) / needed);
    main.style.zoom = String(k.toFixed(3));
    if (main.scrollHeight * k > available + 2) {
      unfit.push((pg.className || '').toString().split(' ')[0] || 'page');
    }
  });
  return unfit;
};

/* Legibility is a hard constraint, not a preference.
 *
 * The floors used to be flat ratios (0.55, 0.50), which meant a section with
 * 10pt type could be scaled to 6.1pt to make its content fit. That is not a
 * fit; it is content the reader cannot use. The floor now comes from the type
 * itself: a block may shrink only until its SMALLEST text reaches the minimum
 * readable size, and no further.
 *
 * The consequence is deliberate. With a fixed canvas you can have at most two
 * of {all the content, readable type, a fixed page count}. Type and page count
 * are design invariants here, so content volume is what has to give -- and it
 * gives through the budgets, where a trim ends on a sentence, rather than
 * through a scale factor that quietly makes everything unreadable.
 */
// 9.5px = 7.1pt on the 1024px canvas (1px = 0.75pt).
//
// Deliberately just BELOW the smallest size anything is authored at (10px), so
// every block keeps a few percent of headroom by construction. Setting the
// floor equal to the authored minimum froze whole sections at zoom 1.00 -- the
// memo's financial and decision sections could not be scaled by even one
// percent and clipped by a handful of pixels.
var MIN_TEXT_PX = 9.5;
var legibilityFloor = el => {
  var min = Infinity;
  el.querySelectorAll('*').forEach(n => {
    if (n.children.length) return;
    if (!n.textContent || !n.textContent.trim()) return;
    var fs = parseFloat(getComputedStyle(n).fontSize) || 0;
    if (fs > 0 && fs < min) min = fs;
  });
  if (!isFinite(min) || min <= 0) return 0;
  return Math.min(1, MIN_TEXT_PX / min);
};

/* The floor actually applied to an element: never below what its type allows. */
var floorFor = (el, base) => Math.max(base, legibilityFloor(el));
var CLIP_FIT_FLOOR = 0.50;

/**
 * Does painted content cross this element's boundary?
 *
 * Deliberately the same test the preflight uses to declare a failure. Every
 * other fitting pass asks a proxy question -- scrollHeight vs clientHeight,
 * height vs the parent row -- and each of those is false in some layout where
 * content is nevertheless printed outside its box. Sections with height:auto
 * grow instead of overflowing; grid children stretch; foreignObjects do not
 * clip. That is how the app could report "Clipped content: v21-thesis" while
 * every fitter believed the page was fine.
 */
// The preflight also names specific descendants that must not cross the
// boundary -- the tail of a table, the decision lens, the bottom line -- because
// those are the parts a reader most obviously loses. Same list, so the fitter
// and the failure report cannot disagree.
var CLIP_CANDIDATES = ['tbody tr:last-child', '.report-sensitivity small', '.report-sensitivity tbody tr:last-child', 'article:last-child', '.report-decision-lens', '.v21-bottom-line'].join(',');
var childContentOverflows = (el, tol = 4) => {
  var er = el.getBoundingClientRect();
  var deep = Array.from(el.querySelectorAll(CLIP_CANDIDATES)).some(n => {
    var r = n.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.bottom > er.bottom + tol;
  });
  if (deep) return true;
  return Array.from(el.children).some(n => {
    var cs = getComputedStyle(n);
    if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    var r = n.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return false;
    return r.bottom > er.bottom + tol || r.right > er.right + tol || r.left < er.left - tol;
  });
};

/**
 * Shrink until nothing crosses a boundary.
 *
 * The final pass, and the only one whose success condition is the same thing
 * the reader sees. Runs after the others so it only has to close whatever gap
 * their proxies missed.
 */
/* The preflight's third failure: a section reaching into the page footer, or
   past the page edge. Same bounds it uses. */
var crossesPageBounds = sec => {
  var page = sec.closest('.report-page, .tp-page, .op-canvas');
  if (!page) return false;
  var pr = page.getBoundingClientRect();
  var r = sec.getBoundingClientRect();
  if (r.width < 1 || r.height < 1) return false;
  if (r.bottom > pr.bottom + 5 || r.right > pr.right + 5) return true;
  var footer = page.querySelector('.report-footer, .tp-footer, footer');
  if (footer && !footer.contains(sec)) {
    var fr = footer.getBoundingClientRect();
    if (fr.height > 0 && r.bottom > fr.top + 3 && r.top < fr.top) return true;
  }
  return false;
};

/* The preflight checks these nodes for page/footer violations, not just
   sections, so the fitter has to start from the same set: a table or a chart
   wrap can be the thing reaching into the footer while its section looks fine. */
var PROTECTED_SEL = 'section, table, .report-pie-wrap, .tp-pie-wrap, ' + '.report-earnings-chart, .tp-chart-row';
export var fitUntilClean = root => {
  if (!root || typeof document === 'undefined') return [];
  var unfit = [];

  // Anything crossing the page edge or the footer gets its owning section
  // scaled, whatever kind of node it is.
  var owners = new Set();
  root.querySelectorAll(PROTECTED_SEL).forEach(n => {
    if (!crossesPageBounds(n)) return;
    var owner = n.closest('.strict-fit') || n.closest('section');
    if (owner) owners.add(owner);
  });
  owners.forEach(sec => {
    var z = parseFloat(sec.style.zoom) || 1;
    var guard = 0;
    var ofloor = floorFor(sec, CLIP_FIT_FLOOR);
    while ((crossesPageBounds(sec) || Array.from(sec.querySelectorAll(PROTECTED_SEL)).some(crossesPageBounds)) && z > ofloor && guard++ < 16) {
      z = Math.max(ofloor, z - 0.035);
      sec.style.zoom = String(z.toFixed(3));
    }
  });
  root.querySelectorAll('.strict-fit').forEach(sec => {
    var z = parseFloat(sec.style.zoom) || 1;
    var guard = 0;
    var bad = () => childContentOverflows(sec) || crossesPageBounds(sec);
    var cfloor = floorFor(sec, CLIP_FIT_FLOOR);
    while (bad() && z > cfloor && guard++ < 16) {
      z = Math.max(cfloor, z - 0.035);
      sec.style.zoom = String(z.toFixed(3));
    }
    if (bad()) {
      unfit.push((sec.className || '').toString().split(' ')[0] || 'section');
    }
  });
  return unfit;
};

/* The page grids whose rows rebalanceLongform sizes. */
var ROW_GRID_SEL = ['.v21-report-p1', '.v21-report-p2', '.v21-report-p3', '.t16-p1-grid', '.t16-p2-grid'].join(',');

/**
 * Allocate page rows by how far each section can legibly compress.
 *
 * rebalanceGridRows divides the page in proportion to how much content each
 * section HAS. That is the wrong currency once type size is a hard floor,
 * because sections are not equally compressible: a block of 14pt prose can give
 * up a third of its height and stay readable, while the financial section --
 * chart, valuation panel and sensitivity matrix, much of it already near the
 * minimum size -- can give up almost nothing. Splitting proportionally hands
 * the incompressible section a row it cannot fit in, and it clips. That is why
 * trimming prose never moved the memo's financial and decision sections.
 *
 * So allocate in the currency that matters: every section first gets the height
 * it needs AT its own legibility floor, and only the slack left over is shared
 * out in proportion to what each would still like. A section can then always
 * reach its row by scaling within the floor.
 *
 * If the floors alone exceed the page, the page genuinely cannot hold this much
 * content at a readable size. That is a content problem, not a layout one, and
 * it is returned to the caller rather than papered over.
 */
export var allocateRowsByFeasibility = root => {
  if (!root || typeof document === 'undefined') return [];
  var over = [];
  root.querySelectorAll(ROW_GRID_SEL).forEach(grid => {
    var kids = Array.from(grid.children).filter(n => n.nodeType === 1);
    if (kids.length < 2) return;
    var cs = getComputedStyle(grid);
    var gap = parseFloat(cs.rowGap) || 0;
    var pad = (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0);
    var available = grid.clientHeight - pad - gap * (kids.length - 1);
    if (available <= 0) return;

    // Natural height means unscaled and unconstrained by the current rows.
    var savedZoom = kids.map(k => k.style.zoom || '');
    var savedRows = grid.style.gridTemplateRows;
    kids.forEach(k => k.style.removeProperty('zoom'));
    grid.style.gridTemplateRows = kids.map(() => 'max-content').join(' ');
    var natural = kids.map(k => Math.max(k.scrollHeight, k.offsetHeight));
    var floors = kids.map(k => Math.max(legibilityFloor(k), SECTION_FIT_FLOOR));
    var needed = natural.map((n, i) => Math.ceil(n * floors[i]) + 2);
    var needSum = needed.reduce((a, b) => a + b, 0);
    var rows;
    if (needSum > available) {
      // Cannot be done legibly. Share the shortfall so the damage is
      // spread rather than dumped on the last section, and report it.
      var k = available / needSum;
      rows = needed.map(d => d * k);
      over.push((grid.className || '').toString().split(' ')[0] || 'grid');
    } else {
      var slack = available - needSum;
      var wantSum = natural.reduce((a, b) => a + b, 0) || 1;
      rows = needed.map((d, i) => Math.min(natural[i] + 2, d + slack * (natural[i] / wantSum)));
      // Anything still unspent goes to the section that wants it most.
      var spent = rows.reduce((a, b) => a + b, 0);
      var left = available - spent;
      if (left > 1) {
        var biggest = 0;
        natural.forEach((n, i) => {
          if (n > natural[biggest]) biggest = i;
        });
        rows[biggest] += left;
      }
    }
    grid.style.gridTemplateRows = rows.map(x => `${x.toFixed(1)}px`).join(' ');
    kids.forEach((k, i) => {
      if (savedZoom[i]) k.style.zoom = savedZoom[i];
    });
    if (!rows.length) grid.style.gridTemplateRows = savedRows;
  });
  return over;
};

/**
 * The single fitting pipeline.
 *
 * There used to be two. The screen ran rebalanceLongform + autoFitSections,
 * and everything else -- the parent fit, the block fit, the foreignObject fit,
 * the page fit -- was wired only into the print path. autoFitSections is
 * precisely the pass that never fires (it asks whether a section overflows
 * ITSELF, and an over-full section in a squeezed row does not), so in practice
 * the on-screen artifact got no fitting at all. That is why the app's own
 * preflight reported clipped content on every section while a harness driving
 * the print path reported clean: they were exercising different code.
 *
 * Order matters: inner boxes first, so an outer pass measures content that has
 * already settled.
 */
export var fitArtifact = root => {
  if (!root || typeof document === 'undefined') return [];
  var unfit = [];
  var step = (fn, label) => {
    try {
      unfit.push(...(fn(root) || []));
    } catch (e) {
      console.warn(`fit ${label}:`, e);
    }
  };
  try {
    rebalanceLongform();
  } catch (e) {
    console.warn('rebalance:', e);
  }
  // Re-cut the rows in the currency that matters before anything is scaled.
  step(allocateRowsByFeasibility, 'rowAllocation');
  step(autoFitBlocks, 'blocks');
  step(autoFitForeignObjects, 'foreignObjects');
  step(autoFitSections, 'sections');
  step(autoFitAgainstParent, 'againstParent');
  step(autoFitPages, 'pages');
  // Last: close whatever the proxy tests above did not catch.
  step(fitUntilClean, 'untilClean');
  return unfit;
};

/**
 * Fit a section to the space it was actually given.
 *
 * autoFitSections asks whether a section's content overflows the section. That
 * misses the case that was losing content: rebalanceGridRows squeezes the grid
 * ROW, and the section then grows taller than the row it sits in. The section
 * itself reports scrollHeight === clientHeight -- perfectly happy -- while the
 * bottom of it is outside the row and off the page. On the CI two-pager the
 * Final Takeaway measured 28.5px inside a 23.0px row, and the last bull point,
 * last bear point and fourth threat body were the part hanging out.
 *
 * So compare each section against its parent's usable height and scale it to
 * fit, rather than trusting the section's own opinion of itself.
 */
export var autoFitAgainstParent = root => {
  if (!root || typeof document === 'undefined') return [];
  var unfit = [];
  root.querySelectorAll('.strict-fit').forEach(sec => {
    var parent = sec.parentElement;
    if (!parent) return;
    var pcs = getComputedStyle(parent);
    var available = parent.clientHeight - (parseFloat(pcs.paddingTop) || 0) - (parseFloat(pcs.paddingBottom) || 0);
    if (!available) return;

    // Measure in layout pixels, not painted ones. getBoundingClientRect is
    // scaled by the PageFit transform on the wrapper while clientHeight is
    // not, so comparing the two made every section look ~14x smaller than
    // its container and this pass never once fired. Reset the zoom, take
    // the natural height, then scale.
    sec.style.removeProperty('zoom');
    var natural = sec.offsetHeight;
    if (!natural || natural <= available + 1) return;
    var k = Math.max(floorFor(sec, SECTION_FIT_FLOOR), (available - 1) / natural);
    sec.style.zoom = String(k.toFixed(3));
    if (sec.offsetHeight * k > available + 2) {
      unfit.push((sec.className || '').toString().split(' ')[0] || 'section');
    }
  });
  return unfit;
};
export var autoFitSections = root => {
  if (!root) return [];
  var unfit = [];
  // .strict-fit is what every fixed-height section actually carries. The
  // selector used to name only .report-section and .tp-section, so the
  // two-pager's sections -- .tp-final, .tp-thesis, .tp-financial -- were
  // never fitted. rebalanceGridRows squeezes their rows when the page is
  // over-full, and with nothing scaling the content inside, the squeezed
  // section simply clipped: that is where the last bull and bear points and
  // the fourth threat's body were going.
  root.querySelectorAll('.report-section, .tp-section, .strict-fit').forEach(sec => {
    sec.style.removeProperty('zoom');
    var box = sec.clientHeight;
    if (!box || sec.scrollHeight <= box + 2) return;

    // `zoom`, not font-size. The v24 sections are built from fixed pixel
    // heights -- tables pinned with height:548px!important, KPI cards with
    // min-height, charts with a fixed viewBox -- so scaling type changes
    // nothing about how tall they are. zoom scales the whole composed block
    // including those pixel values, which is the only lever that actually
    // shrinks this layout without rebuilding it.
    var needed = box / sec.scrollHeight;
    var scale = Math.max(floorFor(sec, SECTION_FIT_FLOOR), needed * 0.995);
    sec.style.zoom = String(scale.toFixed(3));
    if (sec.scrollHeight > sec.clientHeight + 2) {
      unfit.push((sec.className || '').split(' ').find(c => c.startsWith('v21-') || c.startsWith('tp-')) || 'section');
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
export var applyPrintLayout = view => {
  if (typeof document === 'undefined') return null;
  var viewCls = view === 'twopager' ? 'print-twopager' : view === 'memo' ? 'print-report' : 'print-onepager';
  var body = document.body;
  var art = document.querySelector('.dd-artifact');
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
  var chain = [];
  for (var n = art.parentElement; n && n !== body; n = n.parentElement) {
    n.classList.add('dd-print-chain');
    chain.push(n);
  }
  body.classList.add('dd-printing', viewCls);

  // Print changes the available height, so rows and fit are recomputed with
  // the print class active, exactly as the prototype did.
  fitArtifact(art);
  var restored = false;
  var restore = () => {
    if (restored) return;
    restored = true;
    body.classList.remove('dd-printing', 'print-onepager', 'print-twopager', 'print-report');
    chain.forEach(n => n.classList.remove('dd-print-chain'));
    try {
      fitArtifact(art);
    } catch (e) {/* screen only */}
  };
  return restore;
};

/**
 * Save the artifact as a standalone file.
 *
 * "Exactly as it looks on screen" rules out re-rendering it somewhere else.
 * This takes the live DOM of the artifact and every stylesheet the page has
 * loaded, and writes one self-contained HTML file: no network, no fonts to
 * fetch, no server. It opens identically in any browser on any device, and can
 * be printed from there. A PNG was the other option and is a worse one -- the
 * one-pager is an SVG with foreignObject content, which browsers refuse to
 * rasterise to canvas for security reasons.
 */
export var saveArtifact = (view, ticker = 'artifact') => {
  if (typeof document === 'undefined') return false;
  var art = document.querySelector('.dd-artifact');
  if (!art) return false;
  var css = '';
  for (var sheet of Array.from(document.styleSheets)) {
    try {
      css += Array.from(sheet.cssRules).map(r => r.cssText).join('\n') + '\n';
    } catch (e) {
      // A cross-origin sheet cannot be read; ours are same-origin.
    }
  }
  var name = `${ticker}-${view}`.replace(/[^A-Za-z0-9._-]+/g, '-');
  var doc = `<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${name}</title>
<style>
${css}
/* Standalone: no app chrome to hide, and the page is its own background. */
html,body{margin:0;padding:0;background:#fff;}
.dd-artifact{margin:0 auto;}
@page{size:1024px 1536px;margin:0;}
</style>
</head><body class="dd-printing print-${view === 'memo' ? 'report' : view === 'twopager' ? 'twopager' : 'onepager'}">
${art.outerHTML}
</body></html>`;
  var blob = new Blob([doc], {
    type: 'text/html;charset=utf-8'
  });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = `${name}.html`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  return true;
};
export var printArtifact = view => {
  if (typeof document === 'undefined') return;
  var restore = applyPrintLayout(view);
  if (!restore) {
    window.print();
    return;
  }

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
  window.addEventListener('afterprint', restore, {
    once: true
  });
  var onFocus = () => {
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
export var detectOverlaps = page => {
  if (!page || typeof document === 'undefined') return [];
  var SEL = ['.report-card', '.report-callout', '.report-sensitivity', '.report-val-panel',
  // The panel clips, but the blocks inside it were the ones painting over
  // the matrix -- listing only the container hid the real collision.
  '.report-valuation-summary', '.report-target-grid', '.report-matrix-title', '.report-cycle-note', '.report-chart-wrap', '.report-metrics', '.report-table', '.report-threat', '.report-catalyst', '.report-opp', '.report-pool', '.v21-sensitivity', '.v21-bottom-line', '.v21-final-copy', '.report-matrix-title', '.tp-targets', '.tp-fin-bullets', '.tp-chart-row aside', '.tp-cycle', '.nbv-val', '.nbv-chart', '.nbv-target'].join(',');
  /* An element clipped by an ancestor still reports its full layout box, so
     comparing raw rects invents collisions between things the reader never
     sees overlapping. Intersect with every clipping ancestor to get the box
     that actually paints. */
  var visibleRect = el => {
    var r = el.getBoundingClientRect();
    for (var n = el.parentElement; n && n !== page.parentElement; n = n.parentElement) {
      var cs = getComputedStyle(n);
      if (/hidden|clip|auto|scroll/.test(cs.overflow + cs.overflowX + cs.overflowY)) {
        var pr = n.getBoundingClientRect();
        var left = Math.max(r.left, pr.left),
          top = Math.max(r.top, pr.top);
        var right = Math.min(r.right, pr.right),
          bottom = Math.min(r.bottom, pr.bottom);
        r = {
          left,
          top,
          right,
          bottom,
          width: Math.max(0, right - left),
          height: Math.max(0, bottom - top)
        };
      }
    }
    return r;
  };
  var els = Array.from(page.querySelectorAll(SEL)).filter(el => {
    var cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    var r = visibleRect(el);
    return r.width > 8 && r.height > 8;
  });
  var out = [];
  for (var i = 0; i < els.length; i++) {
    for (var j = i + 1; j < els.length; j++) {
      var a = els[i],
        b = els[j];
      // Nesting is not a collision.
      if (a.contains(b) || b.contains(a)) continue;
      var ra = visibleRect(a),
        rb = visibleRect(b);
      var ox = Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left);
      var oy = Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top);
      // A few px of shared edge is normal for adjacent borders.
      if (ox > 4 && oy > 4) {
        var name = el => el.className.toString().trim().split(/\s+/)[0] || el.tagName;
        out.push(`overlap: .${name(a)} and .${name(b)} share ` + `${Math.round(ox)}x${Math.round(oy)}px`);
      }
    }
  }
  return out;
};
export var preflightPages = root => {
  if (!root || typeof document === 'undefined') return [];
  var pages = Array.from(root.querySelectorAll('.op-canvas, .tp-page, .report-page'));
  if (!pages.length) return [];
  var hard = [];
  var qa = {
    failures: []
  };
  var collisions = [];
  try {
    hard = strictClipFailures() || [];
  } catch (e) {
    console.warn('strictClipFailures:', e);
  }
  try {
    qa = collectLayoutQA() || {
      failures: []
    };
  } catch (e) {
    console.warn('collectLayoutQA:', e);
  }
  try {
    pages.forEach(pg => {
      collisions = collisions.concat(detectOverlaps(pg));
    });
  } catch (e) {
    console.warn('detectOverlaps:', e);
  }
  var issues = [...new Set([...(hard || []), ...(qa && qa.failures || []), ...collisions])];
  // The prototype reports per-document, not per-page, so the findings are
  // attached to the first page rather than invented against a page each.
  return pages.map((_page, idx) => ({
    page: idx + 1,
    ok: idx === 0 ? issues.length === 0 : true,
    issues: idx === 0 ? issues : [],
    utilization: null
  }));
};