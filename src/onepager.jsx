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

const { useMemo } = React;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

// Stable pseudo-random in [-1,1] from a string seed. Used for the tiny rotations
// that keep boxes from looking machine-aligned. Seeded so a given page always
// tilts the same way — a poster that reshuffles on every render reads as broken.
function jitter(seed, index = 0) {
    let h = 2166136261;
    const s = `${seed}:${index}`;
    for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return ((h >>> 0) % 2000) / 1000 - 1;
}

// Segment wedge colours, in the order segments arrive. Overridden per style via
// --op-seg-N so a style can carry its own palette without touching this array.
const PIE_COLORS = ['#8fbc6d', '#9dc3e6', '#f5d576', '#c9a0dc', '#f0a07c', '#7fcdc0'];

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
export const ONEPAGER_STYLES = [
    {
        key: 'notebook', label: 'Notebook',
        blurb: 'Hand-drawn on cream paper. Personal, memorable.',
        segmentChart: 'pie', showIcons: true, columns: 2,
    },
    {
        key: 'tearsheet', label: 'Tearsheet',
        blurb: 'Mono, dense, no decoration. For scanning many names.',
        segmentChart: 'bar', showIcons: false, columns: 2,
    },
    {
        key: 'broadsheet', label: 'Broadsheet',
        blurb: 'Serif editorial with rules, not boxes. For circulating.',
        segmentChart: 'pie', showIcons: false, columns: 2,
    },
    {
        key: 'swiss', label: 'Swiss',
        blurb: 'Strict grid, oversized numerals, one accent. For print.',
        segmentChart: 'bar', showIcons: false, columns: 2,
    },
    {
        key: 'deck', label: 'Deck',
        blurb: 'Dark, oversized, single column. For presenting live.',
        segmentChart: 'pie', showIcons: true, columns: 1,
    },
    {
        key: 'ledger', label: 'Ledger',
        blurb: 'Parchment and slab serif, fully ruled. Annual-report register.',
        segmentChart: 'pie', showIcons: true, columns: 2,
    },
];

const STYLE_BY_KEY = ONEPAGER_STYLES.reduce((m, s) => { m[s.key] = s; return m; }, {});

function Icon({ name }) {
    // Deliberately crude single-path glyphs — a hand-drawn page should not carry
    // a polished icon set. Falls back to a dot so an unknown name never breaks.
    const paths = {
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
        flag: 'M6 21V4m0 0h11l-2 4 2 4H6',
    };
    const d = paths[name];
    if (!d) return <span className="op-icon-dot" aria-hidden="true">•</span>;
    return (
        <svg className="op-icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d={d} />
        </svg>
    );
}

function Section({ n, title, accent = 'green', className = '', children, seed }) {
    const tilt = jitter(seed || title, 3) * 0.25;
    return (
        <section
            className={`op-box op-accent-${accent} ${className}`}
            style={{ transform: `rotate(${tilt}deg)` }}
        >
            <h2 className="op-h2">
                {n != null && <span className="op-num">{n}</span>}
                <span className="op-h2-text">{title}</span>
            </h2>
            {children}
        </section>
    );
}

// ---------------------------------------------------------------------------
// charts — plotted from data, never drawn by a model
// ---------------------------------------------------------------------------

function SegmentPie({ segments }) {
    const usable = (segments || []).filter(s => Number(s.share) > 0);
    if (usable.length < 2) return null;

    const total = usable.reduce((sum, s) => sum + Number(s.share), 0);
    const R = 74;
    const CX = 84;
    const CY = 84;

    let angle = -Math.PI / 2; // start at 12 o'clock
    const wedges = usable.map((seg, i) => {
        const frac = Number(seg.share) / total;
        const sweep = frac * Math.PI * 2;
        const x1 = CX + R * Math.cos(angle);
        const y1 = CY + R * Math.sin(angle);
        angle += sweep;
        const x2 = CX + R * Math.cos(angle);
        const y2 = CY + R * Math.sin(angle);
        const mid = angle - sweep / 2;
        const large = sweep > Math.PI ? 1 : 0;
        return {
            key: seg.name || i,
            d: `M${CX},${CY} L${x1.toFixed(1)},${y1.toFixed(1)} A${R},${R} 0 ${large},1 ${x2.toFixed(1)},${y2.toFixed(1)} Z`,
            fill: `var(--op-seg-${i + 1}, ${PIE_COLORS[i % PIE_COLORS.length]})`,
            label: seg.share_label || `${Math.round(frac * 100)}%`,
            lx: CX + R * 0.62 * Math.cos(mid),
            ly: CY + R * 0.62 * Math.sin(mid),
        };
    });

    return (
        <svg className="op-pie" viewBox="0 0 168 168" role="img" aria-label="Segment mix">
            <g filter="url(#op-rough)">
                {wedges.map(w => (
                    <path key={w.key} d={w.d} fill={w.fill} stroke="#2f2b26" strokeWidth="1.4" />
                ))}
            </g>
            {wedges.map(w => (
                <text key={`t-${w.key}`} x={w.lx} y={w.ly} className="op-pie-label"
                      textAnchor="middle" dominantBaseline="middle">
                    {w.label}
                </text>
            ))}
        </svg>
    );
}

// Flat stacked bar — the mix without the ink. Tearsheet and Swiss use this
// because a pie costs 168px of height to say what a 26px bar says, and neither
// style has height to spare.
function SegmentBar({ segments }) {
    const usable = (segments || []).filter(s => Number(s.share) > 0);
    if (!usable.length) return null;
    const total = usable.reduce((sum, s) => sum + Number(s.share), 0);

    return (
        <div className="op-segbar-wrap">
            <div className="op-segbar">
                {usable.map((s, i) => (
                    <div
                        key={s.name || i}
                        className="op-segbar-part"
                        style={{
                            width: `${(Number(s.share) / total) * 100}%`,
                            background: `var(--op-seg-${i + 1}, ${PIE_COLORS[i % PIE_COLORS.length]})`,
                        }}
                        title={`${s.name} ${s.share_label || ''}`}
                    >
                        <span>{s.abbr || s.name}</span>
                    </div>
                ))}
            </div>
            <ul className="op-segbar-key">
                {usable.map((s, i) => (
                    <li key={s.name || i}>
                        <span className="op-swatch"
                              style={{ background: `var(--op-seg-${i + 1}, ${PIE_COLORS[i % PIE_COLORS.length]})` }} />
                        <b>{s.name}</b>
                        <span className="op-segbar-share">{s.share_label || `${Math.round(Number(s.share) / total * 100)}%`}</span>
                    </li>
                ))}
            </ul>
        </div>
    );
}

function EpsChart({ chart }) {
    const pts = (chart?.points || []).filter(p => p && p.year != null && p.eps != null);
    if (pts.length < 3) return null;

    const W = 470;
    const H = 190;
    const PAD = { l: 34, r: 12, t: 12, b: 26 };
    const sorted = [...pts].sort((a, b) => a.year - b.year);

    const years = sorted.map(p => Number(p.year));
    const values = sorted.map(p => Number(p.eps));
    const minY = Math.min(...years);
    const maxY = Math.max(...years);
    const maxV = Math.max(...values, 0);
    // Round the axis top up to a clean step so gridlines land on readable numbers.
    const step = maxV > 40 ? 20 : maxV > 20 ? 10 : 5;
    const top = Math.ceil(maxV / step) * step;

    const sx = y => PAD.l + ((y - minY) / Math.max(1, maxY - minY)) * (W - PAD.l - PAD.r);
    const sy = v => H - PAD.b - (v / Math.max(1, top)) * (H - PAD.t - PAD.b);

    // Split actual from estimate so the forward part can be dashed — the single
    // most important visual honesty cue on the whole page.
    const actual = sorted.filter(p => p.kind !== 'estimate');
    const estimate = sorted.filter(p => p.kind === 'estimate');
    const bridge = actual.length && estimate.length
        ? [actual[actual.length - 1], ...estimate]
        : estimate;

    const line = arr => arr.map((p, i) =>
        `${i === 0 ? 'M' : 'L'}${sx(Number(p.year)).toFixed(1)},${sy(Number(p.eps)).toFixed(1)}`
    ).join(' ');

    const ticks = [];
    for (let v = 0; v <= top; v += step) ticks.push(v);

    return (
        <svg className="op-chart" viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label={chart.label || 'EPS history'}>
            {ticks.map(v => (
                <g key={v}>
                    <line x1={PAD.l} y1={sy(v)} x2={W - PAD.r} y2={sy(v)}
                          className="op-grid" />
                    <text x={PAD.l - 6} y={sy(v)} className="op-axis"
                          textAnchor="end" dominantBaseline="middle">{v}</text>
                </g>
            ))}
            {sorted.filter((_, i) => i % Math.ceil(sorted.length / 9) === 0).map(p => (
                <text key={`x-${p.year}`} x={sx(Number(p.year))} y={H - PAD.b + 14}
                      className="op-axis" textAnchor="middle">
                    {`'${String(p.year).slice(2)}`}
                </text>
            ))}
            <g filter="url(#op-rough)">
                {actual.length > 1 && (
                    <path d={line(actual)} className="op-line" />
                )}
                {bridge.length > 1 && (
                    <path d={line(bridge)} className="op-line op-line-est" />
                )}
            </g>
            {sorted.map(p => (
                <circle key={`d-${p.year}`} cx={sx(Number(p.year))} cy={sy(Number(p.eps))}
                        r="2.6" className={p.kind === 'estimate' ? 'op-dot-est' : 'op-dot'} />
            ))}
            {/* Markers cluster badly on cyclical series — peak, trough and target
                all land in the same few years. Stagger them across three lanes
                and flip the anchor near the edges so labels stay inside the
                viewBox instead of overprinting each other. */}
            {[...(chart.markers || [])]
                .sort((a, b) => Number(a.year) - Number(b.year))
                .map((m, i) => {
                    const x = sx(Number(m.year));
                    const near = sorted.reduce((best, p) =>
                        Math.abs(p.year - m.year) < Math.abs(best.year - m.year) ? p : best, sorted[0]);
                    const y = sy(Number(near.eps));

                    const lane = i % 3;                       // 0 high, 1 low, 2 higher
                    const above = lane !== 1;
                    const reach = lane === 2 ? 34 : 16;
                    const tipY = above ? y - reach : y + reach + 2;
                    const textY = above ? tipY - 4 : tipY + 10;

                    const frac = (x - PAD.l) / (W - PAD.l - PAD.r);
                    const anchor = frac < 0.12 ? 'start' : frac > 0.88 ? 'end' : 'middle';

                    return (
                        <g key={`m-${i}`}>
                            <line x1={x} y1={y} x2={x} y2={tipY} className="op-marker-line" />
                            <text x={x} y={textY} className="op-marker" textAnchor={anchor}>
                                {m.label}
                            </text>
                        </g>
                    );
                })}
        </svg>
    );
}

// ---------------------------------------------------------------------------
// page
// ---------------------------------------------------------------------------

export function OnePager({ data, logoUrl, style = 'notebook' }) {
    if (!data) return null;

    // Unknown style falls back to Notebook rather than rendering unstyled.
    const sty = STYLE_BY_KEY[style] || STYLE_BY_KEY.notebook;

    const {
        ticker = '', company = '', tagline = '',
        at_a_glance: glance = {},
        investment_thesis: thesis = {},
        company_overview: overview = {},
        business_model: model = {},
        opportunities = [],
        financial_snapshot: fin = {},
        signposts = [],
        threats = [],
        takeaway = {},
        meta = {},
    } = data;

    const glanceRows = useMemo(() => ([
        ['Ticker', glance.exchange || ticker],
        ['HQ', glance.hq],
        ['Founded', glance.founded],
        ['Employees', glance.employees],
        ['FY End', glance.fy_end],
        ['Website', glance.website],
    ].filter(([, v]) => v)), [glance, ticker]);

    return (
        <div className="op-sheet" data-op-ticker={ticker} data-op-style={sty.key}>
            {/* One turbulence filter, reused by every frame and chart stroke.
                baseFrequency low + scale small = a pen that wobbles, not a mess. */}
            <svg className="op-defs" aria-hidden="true">
                <defs>
                    <filter id="op-rough">
                        <feTurbulence type="fractalNoise" baseFrequency="0.022"
                                      numOctaves="3" seed="7" result="noise" />
                        <feDisplacementMap in="SourceGraphic" in2="noise"
                                           scale="1.6" xChannelSelector="R" yChannelSelector="G" />
                    </filter>
                </defs>
            </svg>

            <header className="op-header">
                <div className="op-brand">
                    {logoUrl
                        ? <img className="op-logo" src={logoUrl} alt="" />
                        : <div className="op-logo op-logo-ph">{ticker.slice(0, 3)}</div>}
                </div>
                <div className="op-title-wrap">
                    <h1 className="op-title">{company || ticker} {company && `(${ticker})`}</h1>
                    {tagline && <p className="op-tagline">{tagline}</p>}
                </div>
                {glanceRows.length > 0 && (
                    <div className="op-box op-glance op-accent-green">
                        <h2 className="op-h2 op-h2-sm"><span className="op-h2-text">At a Glance</span></h2>
                        <dl>
                            {glanceRows.map(([k, v]) => (
                                <div key={k} className="op-glance-row">
                                    <dt>{k}:</dt><dd>{v}</dd>
                                </div>
                            ))}
                        </dl>
                    </div>
                )}
            </header>

            {meta.is_draft && (
                <div className="op-draft">
                    Draft — assembled from web research, not a curated thesis. Verify before circulating.
                </div>
            )}

            <div className="op-grid">
                <Section n="1" title="Investment Thesis" accent="green" seed={ticker}>
                    {thesis.summary && <p className="op-body">{thesis.summary}</p>}
                    {thesis.core_question && (
                        <p className="op-question">{thesis.core_question}</p>
                    )}
                    <ul className="op-checks">
                        {(thesis.points || []).map((p, i) => (
                            <li key={i}><span className="op-check">☑</span><span>{p}</span></li>
                        ))}
                    </ul>
                </Section>

                <Section n="2" title="Company Overview" accent="blue" seed={ticker}>
                    {overview.summary && <p className="op-body">{overview.summary}</p>}
                    {(overview.segments || []).length > 0 && (
                        <>
                            <h3 className="op-h3">
                                Key Segments
                                {overview.segment_basis && <em> ({overview.segment_basis})</em>}
                            </h3>
                            <div className="op-seg-wrap">
                                {sty.segmentChart === 'bar'
                                    ? <SegmentBar segments={overview.segments} />
                                    : <SegmentPie segments={overview.segments} />}
                                {sty.segmentChart !== 'bar' && <ul className="op-legend">
                                    {overview.segments.map((s, i) => (
                                        <li key={s.name || i}>
                                            <span className="op-swatch"
                                                  style={{ background: `var(--op-seg-${i + 1}, ${PIE_COLORS[i % PIE_COLORS.length]})` }} />
                                            <div>
                                                <strong>{s.name}{s.abbr ? ` (${s.abbr})` : ''}</strong>
                                                {s.description && <span>{s.description}</span>}
                                            </div>
                                        </li>
                                    ))}
                                </ul>}
                            </div>
                        </>
                    )}
                    {overview.footnote && <p className="op-note">{overview.footnote}</p>}
                </Section>

                <Section n="3" title="Business Model" accent="purple" seed={ticker}
                         className="op-span">
                    <div className="op-pools">
                        {(model.profit_pools || []).map((p, i, arr) => (
                            <React.Fragment key={p.name || i}>
                                <div className="op-pool">
                                    <strong>{p.name}</strong>
                                    {p.description && <span>{p.description}</span>}
                                </div>
                                {i < arr.length - 1 && <span className="op-plus">+</span>}
                            </React.Fragment>
                        ))}
                    </div>
                    {model.caption && <p className="op-arrow-note">{model.caption} ⟶</p>}
                </Section>

                <Section n="4" title="Key Opportunities" accent="green" seed={ticker}>
                    <ul className="op-opps">
                        {opportunities.map((o, i) => (
                            <li key={i}>
                                {sty.showIcons && <Icon name={o.icon} />}
                                <div>
                                    <strong>{o.title}</strong>
                                    {o.description && <span>{o.description}</span>}
                                </div>
                            </li>
                        ))}
                    </ul>
                </Section>

                <Section n="5" title="Financial Snapshot" accent="orange" seed={ticker}
                         className="op-span">
                    {fin.period && <span className="op-period">{fin.period}</span>}
                    <div className="op-fin">
                        {/* Metrics and chart share the left column so the chart
                            fills the space beside the (taller) targets stack —
                            otherwise the section opens a hole under the metrics. */}
                        <div className="op-fin-main">
                            <ul className="op-metrics">
                                {(fin.metrics || []).map((m, i) => (
                                    <li key={i}>
                                        <strong>{m.label}:</strong> {m.value}
                                        {m.note && <em> ({m.note})</em>}
                                    </li>
                                ))}
                            </ul>
                            {fin.eps_chart && (
                                <div className="op-chart-wrap">
                                    {fin.eps_chart.label && <h3 className="op-h3">{fin.eps_chart.label}</h3>}
                                    <EpsChart chart={fin.eps_chart} />
                                </div>
                            )}
                        </div>
                        <div className="op-targets">
                            {(fin.mid_cycle_targets || []).length > 0 && (
                                <div className="op-box op-mini">
                                    <h3 className="op-h3">Mid-Cycle Targets</h3>
                                    {fin.mid_cycle_targets.map((t, i) => (
                                        <div key={i} className="op-kv"><span>{t.label}</span><b>{t.value}</b></div>
                                    ))}
                                </div>
                            )}
                            {(fin.valuation || []).length > 0 && (
                                <div className="op-box op-mini op-mini-hl">
                                    <h3 className="op-h3">Valuation</h3>
                                    {fin.valuation.map((t, i) => (
                                        <div key={i} className="op-kv"><span>{t.label}</span><b>{t.value}</b></div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                    {fin.note && <p className="op-note op-note-box">{fin.note}</p>}
                </Section>

                {signposts.length > 0 && (
                    <Section n="6" title="Key Signposts" accent="blue" seed={ticker}
                             className="op-span">
                        <table className="op-table">
                            <thead>
                                <tr><th>Signpost</th><th>Current</th><th>Target</th><th>Why It Matters</th></tr>
                            </thead>
                            <tbody>
                                {signposts.map((s, i) => (
                                    <tr key={i}>
                                        <td><strong>{s.signpost}</strong></td>
                                        <td>{s.current}</td>
                                        <td>{s.target}</td>
                                        <td>{s.why}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </Section>
                )}

                {threats.length > 0 && (
                    <Section n="7" title="Thesis Threats" accent="red" seed={ticker}>
                        <table className="op-table op-table-threats">
                            <thead><tr><th /><th>Watch For</th></tr></thead>
                            <tbody>
                                {threats.map((t, i) => (
                                    <tr key={i}>
                                        <td className="op-threat-name">
                                            {sty.showIcons && <Icon name={t.icon} />}<strong>{t.title}</strong>
                                        </td>
                                        <td>{t.watch_for}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </Section>
                )}

                <Section title="Final Takeaway" accent="gold" seed={ticker}>
                    {takeaway.summary && <p className="op-body">{takeaway.summary}</p>}
                    <div className="op-cases">
                        <div className="op-case op-bull">
                            <h3>↗ Bull Case</h3>
                            <ul>{(takeaway.bull || []).map((b, i) => <li key={i}>{b}</li>)}</ul>
                        </div>
                        <span className="op-vs">vs.</span>
                        <div className="op-case op-bear">
                            <h3>↘ Bear Case</h3>
                            <ul>{(takeaway.bear || []).map((b, i) => <li key={i}>{b}</li>)}</ul>
                        </div>
                    </div>
                </Section>
            </div>

            {takeaway.bottom_line && (
                <footer className="op-bottom">
                    <span className="op-bottom-label">Bottom line:</span>
                    <span className="op-bottom-text">{takeaway.bottom_line}</span>
                </footer>
            )}

            {meta.generated_at && (
                <p className="op-meta">
                    Generated {String(meta.generated_at).slice(0, 10)}
                    {meta.sources?.length ? ` · sources: ${meta.sources.join(', ')}` : ''}
                    {meta.model ? ` · ${meta.model}` : ''}
                </p>
            )}
        </div>
    );
}

export default OnePager;
