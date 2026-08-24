// Investment one-pager renderer.
//
// Draws the JSON from onepager.py as a hand-drawn notebook page. Everything here
// is deterministic: numbers, tables and both charts are computed from the data,
// so the page is exact and identical on every re-render. The hand-drawn quality
// comes from type, an SVG turbulence filter on the frames, and small per-element
// rotations — not from an image model.
//
// The page is deliberately theme-independent. Charlie's Ink/Dusk/Oak/Bloc tokens
// do not reach inside .op-sheet: a printed research page is paper, and it has to
// look the same for whoever it is sent to.

import * as React from 'react';
var {
  useMemo
} = React;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

// Stable pseudo-random in [-1,1] from a string seed. Used for the tiny rotations
// that keep boxes from looking machine-aligned. Seeded so a given page always
// tilts the same way — a poster that reshuffles on every render reads as broken.
function jitter(seed, index = 0) {
  var h = 2166136261;
  var s = `${seed}:${index}`;
  for (var i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % 2000 / 1000 - 1;
}

// Segment wedge colours, in the order segments arrive. Overridden per style via
// --op-seg-N so a style can carry its own palette without touching this array.
var PIE_COLORS = ['#8fbc6d', '#9dc3e6', '#f5d576', '#c9a0dc', '#f0a07c', '#7fcdc0'];

// ---------------------------------------------------------------------------
// styles
// ---------------------------------------------------------------------------
// A style is mostly CSS (see onepager.css), but three decisions are structural
// and have to happen here — a stylesheet cannot swap a pie for a bar or drop an
// icon column. Everything else stays in the token blocks.
//
//   segmentChart  'pie' | 'bar'   how the business mix is drawn
//   showIcons     bool            icon column on opportunities/threats
//   columns       1 | 2           top-level grid
export var ONEPAGER_STYLES = [{
  key: 'notebook',
  label: 'Notebook',
  blurb: 'Hand-drawn on cream paper. Personal, memorable.',
  segmentChart: 'pie',
  showIcons: true,
  columns: 2
}, {
  key: 'tearsheet',
  label: 'Tearsheet',
  blurb: 'Mono, dense, no decoration. For scanning many names.',
  segmentChart: 'bar',
  showIcons: false,
  columns: 2
}, {
  key: 'broadsheet',
  label: 'Broadsheet',
  blurb: 'Serif editorial with rules, not boxes. For circulating.',
  segmentChart: 'pie',
  showIcons: false,
  columns: 2
}, {
  key: 'swiss',
  label: 'Swiss',
  blurb: 'Strict grid, oversized numerals, one accent. For print.',
  segmentChart: 'bar',
  showIcons: false,
  columns: 2
}, {
  key: 'deck',
  label: 'Deck',
  blurb: 'Dark, oversized, single column. For presenting live.',
  segmentChart: 'pie',
  showIcons: true,
  columns: 1
}, {
  key: 'ledger',
  label: 'Ledger',
  blurb: 'Parchment and slab serif, fully ruled. Annual-report register.',
  segmentChart: 'pie',
  showIcons: true,
  columns: 2
}];
var STYLE_BY_KEY = ONEPAGER_STYLES.reduce((m, s) => {
  m[s.key] = s;
  return m;
}, {});
function Icon({
  name
}) {
  // Deliberately crude single-path glyphs — a hand-drawn page should not carry
  // a polished icon set. Falls back to a dot so an unknown name never breaks.
  var paths = {
    leaf: 'M4 20c0-8 6-14 16-16 0 10-6 16-16 16zm0 0c4-4 8-6 12-8',
    people: 'M8 11a3 3 0 100-6 3 3 0 000 6zm8 0a3 3 0 100-6 3 3 0 000 6zM2 21c0-4 3-6 6-6s6 2 6 6M14 21c0-4 2-6 5-6s5 2 5 6',
    cycle: 'M4 12a8 8 0 0114-5M20 12a8 8 0 01-14 5M18 4v3h-3M6 20v-3h3',
    globe: 'M12 21a9 9 0 100-18 9 9 0 000 18zM3 12h18M12 3c3 4 3 14 0 18M12 3c-3 4-3 14 0 18',
    bank: 'M3 10h18L12 4 3 10zm2 0v8m4-8v8m6-8v8m4-8v8M3 20h18',
    chip: 'M7 7h10v10H7zM4 10h3M4 14h3M17 10h3M17 14h3M10 4v3M14 4v3M10 17v3M14 17v3',
    chart: 'M4 20V6m0 14h16M8 20v-6m4 6V9m4 11v-8',
    truck: 'M2 16V7h11v9M13 10h4l4 4v2h-8M6 20a2 2 0 100-4 2 2 0 000 4zm11 0a2 2 0 100-4 2 2 0 000 4z',
    wrench: 'M20 5a5 5 0 01-6.5 6.5L5 20l-2-2 8.5-8.5A5 5 0 0119 4l-3 3 2 2 3-3z',
    shield: 'M12 3l8 3v6c0 5-4 8-8 9-4-1-8-4-8-9V6l8-3z',
    cloud: 'M6 18a4 4 0 010-8 6 6 0 0111-2 4 4 0 011 8H6zM8 21l1-2m3 2l1-2m3 2l1-2',
    chart_down: 'M4 6v14h16M8 10l4 5 3-3 4 5',
    gauge: 'M12 20a8 8 0 100-16 8 8 0 000 16zM12 12l4-3M12 12v5',
    gear: 'M12 15a3 3 0 100-6 3 3 0 000 6zM12 3v3m0 12v3M3 12h3m12 0h3M5.5 5.5l2 2m9 9l2 2m-13 0l2-2m9-9l2-2',
    scale: 'M12 4v16M6 8h12M6 8L3 15h6L6 8zm12 0l-3 7h6l-3-7zM8 20h8',
    flag: 'M6 21V4m0 0h11l-2 4 2 4H6'
  };
  var d = paths[name];
  if (!d) return /*#__PURE__*/React.createElement("span", {
    className: "op-icon-dot",
    "aria-hidden": "true"
  }, "\u2022");
  return /*#__PURE__*/React.createElement("svg", {
    className: "op-icon",
    viewBox: "0 0 24 24",
    "aria-hidden": "true"
  }, /*#__PURE__*/React.createElement("path", {
    d: d
  }));
}
function Section({
  n,
  title,
  accent = 'green',
  className = '',
  children,
  seed
}) {
  var tilt = jitter(seed || title, 3) * 0.25;
  return /*#__PURE__*/React.createElement("section", {
    className: `op-box op-accent-${accent} ${className}`,
    style: {
      transform: `rotate(${tilt}deg)`
    }
  }, /*#__PURE__*/React.createElement("h2", {
    className: "op-h2"
  }, n != null && /*#__PURE__*/React.createElement("span", {
    className: "op-num"
  }, n), /*#__PURE__*/React.createElement("span", {
    className: "op-h2-text"
  }, title)), children);
}

// ---------------------------------------------------------------------------
// charts — plotted from data, never drawn by a model
// ---------------------------------------------------------------------------

function SegmentPie({
  segments
}) {
  var usable = (segments || []).filter(s => Number(s.share) > 0);
  if (usable.length < 2) return null;
  var total = usable.reduce((sum, s) => sum + Number(s.share), 0);
  var R = 74;
  var CX = 84;
  var CY = 84;
  var angle = -Math.PI / 2; // start at 12 o'clock
  var wedges = usable.map((seg, i) => {
    var frac = Number(seg.share) / total;
    var sweep = frac * Math.PI * 2;
    var x1 = CX + R * Math.cos(angle);
    var y1 = CY + R * Math.sin(angle);
    angle += sweep;
    var x2 = CX + R * Math.cos(angle);
    var y2 = CY + R * Math.sin(angle);
    var mid = angle - sweep / 2;
    var large = sweep > Math.PI ? 1 : 0;
    return {
      key: seg.name || i,
      d: `M${CX},${CY} L${x1.toFixed(1)},${y1.toFixed(1)} A${R},${R} 0 ${large},1 ${x2.toFixed(1)},${y2.toFixed(1)} Z`,
      fill: `var(--op-seg-${i + 1}, ${PIE_COLORS[i % PIE_COLORS.length]})`,
      label: seg.share_label || `${Math.round(frac * 100)}%`,
      lx: CX + R * 0.62 * Math.cos(mid),
      ly: CY + R * 0.62 * Math.sin(mid)
    };
  });
  return /*#__PURE__*/React.createElement("svg", {
    className: "op-pie",
    viewBox: "0 0 168 168",
    role: "img",
    "aria-label": "Segment mix"
  }, /*#__PURE__*/React.createElement("g", {
    filter: "url(#op-rough)"
  }, wedges.map(w => /*#__PURE__*/React.createElement("path", {
    key: w.key,
    d: w.d,
    fill: w.fill,
    stroke: "#2f2b26",
    strokeWidth: "1.4"
  }))), wedges.map(w => /*#__PURE__*/React.createElement("text", {
    key: `t-${w.key}`,
    x: w.lx,
    y: w.ly,
    className: "op-pie-label",
    textAnchor: "middle",
    dominantBaseline: "middle"
  }, w.label)));
}

// Flat stacked bar — the mix without the ink. Tearsheet and Swiss use this
// because a pie costs 168px of height to say what a 26px bar says, and neither
// style has height to spare.
function SegmentBar({
  segments
}) {
  var usable = (segments || []).filter(s => Number(s.share) > 0);
  if (!usable.length) return null;
  var total = usable.reduce((sum, s) => sum + Number(s.share), 0);
  return /*#__PURE__*/React.createElement("div", {
    className: "op-segbar-wrap"
  }, /*#__PURE__*/React.createElement("div", {
    className: "op-segbar"
  }, usable.map((s, i) => /*#__PURE__*/React.createElement("div", {
    key: s.name || i,
    className: "op-segbar-part",
    style: {
      width: `${Number(s.share) / total * 100}%`,
      background: `var(--op-seg-${i + 1}, ${PIE_COLORS[i % PIE_COLORS.length]})`
    },
    title: `${s.name} ${s.share_label || ''}`
  }, /*#__PURE__*/React.createElement("span", null, s.abbr || s.name)))), /*#__PURE__*/React.createElement("ul", {
    className: "op-segbar-key"
  }, usable.map((s, i) => /*#__PURE__*/React.createElement("li", {
    key: s.name || i
  }, /*#__PURE__*/React.createElement("span", {
    className: "op-swatch",
    style: {
      background: `var(--op-seg-${i + 1}, ${PIE_COLORS[i % PIE_COLORS.length]})`
    }
  }), /*#__PURE__*/React.createElement("b", null, s.name), /*#__PURE__*/React.createElement("span", {
    className: "op-segbar-share"
  }, s.share_label || `${Math.round(Number(s.share) / total * 100)}%`)))));
}
function EpsChart({
  chart
}) {
  var pts = (chart?.points || []).filter(p => p && p.year != null && p.eps != null);
  if (pts.length < 3) return null;
  var W = 470;
  var H = 190;
  var PAD = {
    l: 34,
    r: 12,
    t: 12,
    b: 26
  };
  var sorted = [...pts].sort((a, b) => a.year - b.year);
  var years = sorted.map(p => Number(p.year));
  var values = sorted.map(p => Number(p.eps));
  var minY = Math.min(...years);
  var maxY = Math.max(...years);
  var maxV = Math.max(...values, 0);
  // Round the axis top up to a clean step so gridlines land on readable numbers.
  var step = maxV > 40 ? 20 : maxV > 20 ? 10 : 5;
  var top = Math.ceil(maxV / step) * step;
  var sx = y => PAD.l + (y - minY) / Math.max(1, maxY - minY) * (W - PAD.l - PAD.r);
  var sy = v => H - PAD.b - v / Math.max(1, top) * (H - PAD.t - PAD.b);

  // Split actual from estimate so the forward part can be dashed — the single
  // most important visual honesty cue on the whole page.
  var actual = sorted.filter(p => p.kind !== 'estimate');
  var estimate = sorted.filter(p => p.kind === 'estimate');
  var bridge = actual.length && estimate.length ? [actual[actual.length - 1], ...estimate] : estimate;
  var line = arr => arr.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(Number(p.year)).toFixed(1)},${sy(Number(p.eps)).toFixed(1)}`).join(' ');
  var ticks = [];
  for (var v = 0; v <= top; v += step) ticks.push(v);
  return /*#__PURE__*/React.createElement("svg", {
    className: "op-chart",
    viewBox: `0 0 ${W} ${H}`,
    role: "img",
    "aria-label": chart.label || 'EPS history'
  }, ticks.map(v => /*#__PURE__*/React.createElement("g", {
    key: v
  }, /*#__PURE__*/React.createElement("line", {
    x1: PAD.l,
    y1: sy(v),
    x2: W - PAD.r,
    y2: sy(v),
    className: "op-grid"
  }), /*#__PURE__*/React.createElement("text", {
    x: PAD.l - 6,
    y: sy(v),
    className: "op-axis",
    textAnchor: "end",
    dominantBaseline: "middle"
  }, v))), sorted.filter((_, i) => i % Math.ceil(sorted.length / 9) === 0).map(p => /*#__PURE__*/React.createElement("text", {
    key: `x-${p.year}`,
    x: sx(Number(p.year)),
    y: H - PAD.b + 14,
    className: "op-axis",
    textAnchor: "middle"
  }, `'${String(p.year).slice(2)}`)), /*#__PURE__*/React.createElement("g", {
    filter: "url(#op-rough)"
  }, actual.length > 1 && /*#__PURE__*/React.createElement("path", {
    d: line(actual),
    className: "op-line"
  }), bridge.length > 1 && /*#__PURE__*/React.createElement("path", {
    d: line(bridge),
    className: "op-line op-line-est"
  })), sorted.map(p => /*#__PURE__*/React.createElement("circle", {
    key: `d-${p.year}`,
    cx: sx(Number(p.year)),
    cy: sy(Number(p.eps)),
    r: "2.6",
    className: p.kind === 'estimate' ? 'op-dot-est' : 'op-dot'
  })), [...(chart.markers || [])].sort((a, b) => Number(a.year) - Number(b.year)).map((m, i) => {
    var x = sx(Number(m.year));
    var near = sorted.reduce((best, p) => Math.abs(p.year - m.year) < Math.abs(best.year - m.year) ? p : best, sorted[0]);
    var y = sy(Number(near.eps));
    var lane = i % 3; // 0 high, 1 low, 2 higher
    var above = lane !== 1;
    var reach = lane === 2 ? 34 : 16;
    var tipY = above ? y - reach : y + reach + 2;
    var textY = above ? tipY - 4 : tipY + 10;
    var frac = (x - PAD.l) / (W - PAD.l - PAD.r);
    var anchor = frac < 0.12 ? 'start' : frac > 0.88 ? 'end' : 'middle';
    return /*#__PURE__*/React.createElement("g", {
      key: `m-${i}`
    }, /*#__PURE__*/React.createElement("line", {
      x1: x,
      y1: y,
      x2: x,
      y2: tipY,
      className: "op-marker-line"
    }), /*#__PURE__*/React.createElement("text", {
      x: x,
      y: textY,
      className: "op-marker",
      textAnchor: anchor
    }, m.label));
  }));
}

// ---------------------------------------------------------------------------
// page
// ---------------------------------------------------------------------------

export function OnePager({
  data,
  logoUrl,
  style = 'notebook'
}) {
  if (!data) return null;

  // Unknown style falls back to Notebook rather than rendering unstyled.
  var sty = STYLE_BY_KEY[style] || STYLE_BY_KEY.notebook;
  var {
    ticker = '',
    company = '',
    tagline = '',
    at_a_glance: glance = {},
    investment_thesis: thesis = {},
    company_overview: overview = {},
    business_model: model = {},
    opportunities = [],
    financial_snapshot: fin = {},
    signposts = [],
    threats = [],
    takeaway = {},
    meta = {}
  } = data;
  var glanceRows = useMemo(() => [['Ticker', glance.exchange || ticker], ['HQ', glance.hq], ['Founded', glance.founded], ['Employees', glance.employees], ['FY End', glance.fy_end], ['Website', glance.website]].filter(([, v]) => v), [glance, ticker]);
  return /*#__PURE__*/React.createElement("div", {
    className: "op-sheet",
    "data-op-ticker": ticker,
    "data-op-style": sty.key
  }, /*#__PURE__*/React.createElement("svg", {
    className: "op-defs",
    "aria-hidden": "true"
  }, /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("filter", {
    id: "op-rough"
  }, /*#__PURE__*/React.createElement("feTurbulence", {
    type: "fractalNoise",
    baseFrequency: "0.022",
    numOctaves: "3",
    seed: "7",
    result: "noise"
  }), /*#__PURE__*/React.createElement("feDisplacementMap", {
    in: "SourceGraphic",
    in2: "noise",
    scale: "1.6",
    xChannelSelector: "R",
    yChannelSelector: "G"
  })))), /*#__PURE__*/React.createElement("header", {
    className: "op-header"
  }, /*#__PURE__*/React.createElement("div", {
    className: "op-brand"
  }, logoUrl ? /*#__PURE__*/React.createElement("img", {
    className: "op-logo",
    src: logoUrl,
    alt: ""
  }) : /*#__PURE__*/React.createElement("div", {
    className: "op-logo op-logo-ph"
  }, ticker.slice(0, 3))), /*#__PURE__*/React.createElement("div", {
    className: "op-title-wrap"
  }, /*#__PURE__*/React.createElement("h1", {
    className: "op-title"
  }, company || ticker, " ", company && `(${ticker})`), tagline && /*#__PURE__*/React.createElement("p", {
    className: "op-tagline"
  }, tagline)), glanceRows.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "op-box op-glance op-accent-green"
  }, /*#__PURE__*/React.createElement("h2", {
    className: "op-h2 op-h2-sm"
  }, /*#__PURE__*/React.createElement("span", {
    className: "op-h2-text"
  }, "At a Glance")), /*#__PURE__*/React.createElement("dl", null, glanceRows.map(([k, v]) => /*#__PURE__*/React.createElement("div", {
    key: k,
    className: "op-glance-row"
  }, /*#__PURE__*/React.createElement("dt", null, k, ":"), /*#__PURE__*/React.createElement("dd", null, v)))))), meta.is_draft && /*#__PURE__*/React.createElement("div", {
    className: "op-draft"
  }, "Draft \u2014 assembled from web research, not a curated thesis. Verify before circulating."), /*#__PURE__*/React.createElement("div", {
    className: "op-grid"
  }, /*#__PURE__*/React.createElement(Section, {
    n: "1",
    title: "Investment Thesis",
    accent: "green",
    seed: ticker
  }, thesis.summary && /*#__PURE__*/React.createElement("p", {
    className: "op-body"
  }, thesis.summary), thesis.core_question && /*#__PURE__*/React.createElement("p", {
    className: "op-question"
  }, thesis.core_question), /*#__PURE__*/React.createElement("ul", {
    className: "op-checks"
  }, (thesis.points || []).map((p, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, /*#__PURE__*/React.createElement("span", {
    className: "op-check"
  }, "\u2611"), /*#__PURE__*/React.createElement("span", null, p))))), /*#__PURE__*/React.createElement(Section, {
    n: "2",
    title: "Company Overview",
    accent: "blue",
    seed: ticker
  }, overview.summary && /*#__PURE__*/React.createElement("p", {
    className: "op-body"
  }, overview.summary), (overview.segments || []).length > 0 && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h3", {
    className: "op-h3"
  }, "Key Segments", overview.segment_basis && /*#__PURE__*/React.createElement("em", null, " (", overview.segment_basis, ")")), /*#__PURE__*/React.createElement("div", {
    className: "op-seg-wrap"
  }, sty.segmentChart === 'bar' ? /*#__PURE__*/React.createElement(SegmentBar, {
    segments: overview.segments
  }) : /*#__PURE__*/React.createElement(SegmentPie, {
    segments: overview.segments
  }), sty.segmentChart !== 'bar' && /*#__PURE__*/React.createElement("ul", {
    className: "op-legend"
  }, overview.segments.map((s, i) => /*#__PURE__*/React.createElement("li", {
    key: s.name || i
  }, /*#__PURE__*/React.createElement("span", {
    className: "op-swatch",
    style: {
      background: `var(--op-seg-${i + 1}, ${PIE_COLORS[i % PIE_COLORS.length]})`
    }
  }), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, s.name, s.abbr ? ` (${s.abbr})` : ''), s.description && /*#__PURE__*/React.createElement("span", null, s.description))))))), overview.footnote && /*#__PURE__*/React.createElement("p", {
    className: "op-note"
  }, overview.footnote)), /*#__PURE__*/React.createElement(Section, {
    n: "3",
    title: "Business Model",
    accent: "purple",
    seed: ticker,
    className: "op-span"
  }, /*#__PURE__*/React.createElement("div", {
    className: "op-pools"
  }, (model.profit_pools || []).map((p, i, arr) => /*#__PURE__*/React.createElement(React.Fragment, {
    key: p.name || i
  }, /*#__PURE__*/React.createElement("div", {
    className: "op-pool"
  }, /*#__PURE__*/React.createElement("strong", null, p.name), p.description && /*#__PURE__*/React.createElement("span", null, p.description)), i < arr.length - 1 && /*#__PURE__*/React.createElement("span", {
    className: "op-plus"
  }, "+")))), model.caption && /*#__PURE__*/React.createElement("p", {
    className: "op-arrow-note"
  }, model.caption, " \u27F6")), /*#__PURE__*/React.createElement(Section, {
    n: "4",
    title: "Key Opportunities",
    accent: "green",
    seed: ticker
  }, /*#__PURE__*/React.createElement("ul", {
    className: "op-opps"
  }, opportunities.map((o, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, sty.showIcons && /*#__PURE__*/React.createElement(Icon, {
    name: o.icon
  }), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, o.title), o.description && /*#__PURE__*/React.createElement("span", null, o.description)))))), /*#__PURE__*/React.createElement(Section, {
    n: "5",
    title: "Financial Snapshot",
    accent: "orange",
    seed: ticker,
    className: "op-span"
  }, fin.period && /*#__PURE__*/React.createElement("span", {
    className: "op-period"
  }, fin.period), /*#__PURE__*/React.createElement("div", {
    className: "op-fin"
  }, /*#__PURE__*/React.createElement("div", {
    className: "op-fin-main"
  }, /*#__PURE__*/React.createElement("ul", {
    className: "op-metrics"
  }, (fin.metrics || []).map((m, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, /*#__PURE__*/React.createElement("strong", null, m.label, ":"), " ", m.value, m.note && /*#__PURE__*/React.createElement("em", null, " (", m.note, ")")))), fin.eps_chart && /*#__PURE__*/React.createElement("div", {
    className: "op-chart-wrap"
  }, fin.eps_chart.label && /*#__PURE__*/React.createElement("h3", {
    className: "op-h3"
  }, fin.eps_chart.label), /*#__PURE__*/React.createElement(EpsChart, {
    chart: fin.eps_chart
  }))), /*#__PURE__*/React.createElement("div", {
    className: "op-targets"
  }, (fin.mid_cycle_targets || []).length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "op-box op-mini"
  }, /*#__PURE__*/React.createElement("h3", {
    className: "op-h3"
  }, "Mid-Cycle Targets"), fin.mid_cycle_targets.map((t, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "op-kv"
  }, /*#__PURE__*/React.createElement("span", null, t.label), /*#__PURE__*/React.createElement("b", null, t.value)))), (fin.valuation || []).length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "op-box op-mini op-mini-hl"
  }, /*#__PURE__*/React.createElement("h3", {
    className: "op-h3"
  }, "Valuation"), fin.valuation.map((t, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    className: "op-kv"
  }, /*#__PURE__*/React.createElement("span", null, t.label), /*#__PURE__*/React.createElement("b", null, t.value)))))), fin.note && /*#__PURE__*/React.createElement("p", {
    className: "op-note op-note-box"
  }, fin.note)), signposts.length > 0 && /*#__PURE__*/React.createElement(Section, {
    n: "6",
    title: "Key Signposts",
    accent: "blue",
    seed: ticker,
    className: "op-span"
  }, /*#__PURE__*/React.createElement("table", {
    className: "op-table"
  }, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "Signpost"), /*#__PURE__*/React.createElement("th", null, "Current"), /*#__PURE__*/React.createElement("th", null, "Target"), /*#__PURE__*/React.createElement("th", null, "Why It Matters"))), /*#__PURE__*/React.createElement("tbody", null, signposts.map((s, i) => /*#__PURE__*/React.createElement("tr", {
    key: i
  }, /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("strong", null, s.signpost)), /*#__PURE__*/React.createElement("td", null, s.current), /*#__PURE__*/React.createElement("td", null, s.target), /*#__PURE__*/React.createElement("td", null, s.why)))))), threats.length > 0 && /*#__PURE__*/React.createElement(Section, {
    n: "7",
    title: "Thesis Threats",
    accent: "red",
    seed: ticker
  }, /*#__PURE__*/React.createElement("table", {
    className: "op-table op-table-threats"
  }, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null), /*#__PURE__*/React.createElement("th", null, "Watch For"))), /*#__PURE__*/React.createElement("tbody", null, threats.map((t, i) => /*#__PURE__*/React.createElement("tr", {
    key: i
  }, /*#__PURE__*/React.createElement("td", {
    className: "op-threat-name"
  }, sty.showIcons && /*#__PURE__*/React.createElement(Icon, {
    name: t.icon
  }), /*#__PURE__*/React.createElement("strong", null, t.title)), /*#__PURE__*/React.createElement("td", null, t.watch_for)))))), /*#__PURE__*/React.createElement(Section, {
    title: "Final Takeaway",
    accent: "gold",
    seed: ticker
  }, takeaway.summary && /*#__PURE__*/React.createElement("p", {
    className: "op-body"
  }, takeaway.summary), /*#__PURE__*/React.createElement("div", {
    className: "op-cases"
  }, /*#__PURE__*/React.createElement("div", {
    className: "op-case op-bull"
  }, /*#__PURE__*/React.createElement("h3", null, "\u2197 Bull Case"), /*#__PURE__*/React.createElement("ul", null, (takeaway.bull || []).map((b, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, b)))), /*#__PURE__*/React.createElement("span", {
    className: "op-vs"
  }, "vs."), /*#__PURE__*/React.createElement("div", {
    className: "op-case op-bear"
  }, /*#__PURE__*/React.createElement("h3", null, "\u2198 Bear Case"), /*#__PURE__*/React.createElement("ul", null, (takeaway.bear || []).map((b, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, b))))))), takeaway.bottom_line && /*#__PURE__*/React.createElement("footer", {
    className: "op-bottom"
  }, /*#__PURE__*/React.createElement("span", {
    className: "op-bottom-label"
  }, "Bottom line:"), /*#__PURE__*/React.createElement("span", {
    className: "op-bottom-text"
  }, takeaway.bottom_line)), meta.generated_at && /*#__PURE__*/React.createElement("p", {
    className: "op-meta"
  }, "Generated ", String(meta.generated_at).slice(0, 10), meta.sources?.length ? ` · sources: ${meta.sources.join(', ')}` : '', meta.model ? ` · ${meta.model}` : ''));
}
export default OnePager;