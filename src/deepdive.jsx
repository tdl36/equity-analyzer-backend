// Deep Dive renderers — one canonical research run, three distinct artifacts.
//
// Ported from the Investment Research Studio v24 prototype, calibrated against
// the reviewed DE v29 PDFs (the "best proven" baseline in the handoff).
//
// WHY THE CANVAS IS FIXED
// ---------------------------------------------------------------------------
// Every page is exactly 1024x1536 CSS px, which is 768x1152pt at 96dpi — the
// dimensions of the v29 renders. These are *composed* pages, not documents that
// reflow: the whole point of the one-pager is that a known amount of content
// lands in a known place. Reflowing it to the viewport would produce a
// different artifact at every window width and make the print output
// unpredictable, which is exactly the failure the prototype spent iterations
// escaping.
//
// On screen the page is SCALED to fit its container; for print it is rendered
// at true size. Never solve a fit problem by shrinking type — the handoff is
// explicit that extra pages exist to buy larger fonts, not more prose.

import * as React from 'react';

const { useState, useEffect, useRef, useCallback } = React;

export const PAGE_W = 1024;
export const PAGE_H = 1536;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

const txt = (v, fallback = '') => {
    if (v === null || v === undefined) return fallback;
    if (typeof v === 'string') return v;
    if (typeof v === 'number') return String(v);
    return fallback;
};

const arr = (v) => (Array.isArray(v) ? v : []);

// Deterministic jitter so the hand-drawn look is stable across renders.
// Math.random() here would make every re-render wobble and every screenshot
// diff fail, which defeats visual regression testing.
const jitter = (seed, spread = 1) => {
    const x = Math.sin(seed * 12.9898) * 43758.5453;
    return ((x - Math.floor(x)) - 0.5) * 2 * spread;
};

/** Scales a fixed-size page down to fit its container width. */
export const PageFit = ({ width = PAGE_W, height = PAGE_H, className = '', children }) => {
    const wrapRef = useRef(null);
    const [scale, setScale] = useState(1);
    const lastWidth = useRef(0);

    useEffect(() => {
        const el = wrapRef.current;
        if (!el) return;
        let frame = null;

        const apply = () => {
            const w = el.clientWidth || 0;
            if (!w) return;
            setScale(Math.min(1, w / width));
        };
        const schedule = () => {
            if (frame) cancelAnimationFrame(frame);
            frame = requestAnimationFrame(apply);
        };

        // Observe only the wrapper's WIDTH. Observing an element whose height
        // this component then changes produces a ResizeObserver feedback loop —
        // the browser reports it as an error and it looks like a crash.
        const ro = new ResizeObserver((entries) => {
            const w = entries?.[0]?.contentRect?.width ?? el.clientWidth ?? 0;
            if (Math.abs(w - lastWidth.current) < 0.5) return;
            lastWidth.current = w;
            schedule();
        });
        ro.observe(el);
        apply();
        return () => { ro.disconnect(); if (frame) cancelAnimationFrame(frame); };
    }, [width]);

    return (
        <div ref={wrapRef} className={`dd-fit ${className}`}
             style={{ height: height * scale }}>
            <div className="dd-fit-inner"
                 style={{ width, height, transform: `scale(${scale})` }}>
                {children}
            </div>
        </div>
    );
};

// ---------------------------------------------------------------------------
// shared visual primitives
// ---------------------------------------------------------------------------

const PIE_COLORS = ['#7aa86f', '#8fb3d9', '#e3c169', '#c58fb0', '#8f8fc5'];

/** Segment pie. Falls back to nothing when shares do not close to ~100. */
export const SegmentPie = ({ segments, size = 210, hand = false }) => {
    const segs = arr(segments).filter(s => Number(s?.mix_numeric) > 0);
    const total = segs.reduce((a, s) => a + Number(s.mix_numeric), 0);
    if (!segs.length || total < 85 || total > 115) return null;

    const r = size / 2 - 4;
    const cx = size / 2, cy = size / 2;
    let angle = -Math.PI / 2;

    return (
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="dd-pie">
            {segs.map((s, i) => {
                const frac = Number(s.mix_numeric) / total;
                const end = angle + frac * Math.PI * 2;
                const large = frac > 0.5 ? 1 : 0;
                const wob = hand ? jitter(i + 1, 1.6) : 0;
                const x1 = cx + Math.cos(angle) * (r + wob);
                const y1 = cy + Math.sin(angle) * (r + wob);
                const x2 = cx + Math.cos(end) * (r + wob);
                const y2 = cy + Math.sin(end) * (r + wob);
                const mid = angle + (end - angle) / 2;
                const lx = cx + Math.cos(mid) * r * 0.6;
                const ly = cy + Math.sin(mid) * r * 0.6;
                const path = `M ${cx} ${cy} L ${x1} ${y1} A ${r + wob} ${r + wob} 0 ${large} 1 ${x2} ${y2} Z`;
                angle = end;
                return (
                    <g key={i}>
                        <path d={path} fill={PIE_COLORS[i % PIE_COLORS.length]}
                              stroke="#3a3a3a" strokeWidth={hand ? 1.4 : 0.8}
                              opacity={hand ? 0.85 : 1} />
                        <text x={lx} y={ly} className="dd-pie-label"
                              textAnchor="middle" dominantBaseline="middle">
                            {txt(s.mix)}
                        </text>
                    </g>
                );
            })}
        </svg>
    );
};

/**
 * Earnings-cycle line chart.
 *
 * Label placement is deliberate: v29 let the "2030 target" and "trough"
 * annotations sit on top of the plotted line. Annotations are offset away from
 * their point here, and the right margin is reserved so end-of-series labels
 * have somewhere to go instead of overlapping the data.
 */
export const EarningsChart = ({ history, width = 620, height = 250, hand = false }) => {
    const points = arr(history?.points).filter(p => Number.isFinite(Number(p?.value)));
    if (points.length < 3) return null;

    const padL = 44, padR = 150, padT = 26, padB = 30;
    const values = points.map(p => Number(p.value));
    const maxV = Math.max(...values) * 1.15;
    const minV = Math.min(0, Math.min(...values));
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    const x = (i) => padL + (plotW * i) / Math.max(1, points.length - 1);
    const y = (v) => padT + plotH - ((v - minV) / (maxV - minV || 1)) * plotH;

    const line = points.map((p, i) =>
        `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${(y(Number(p.value)) + (hand ? jitter(i, 1.1) : 0)).toFixed(1)}`
    ).join(' ');

    const ticks = [minV, (minV + maxV) / 2, maxV].map(v => Math.round(v));

    return (
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="dd-chart">
            {ticks.map((t, i) => (
                <g key={i}>
                    <line x1={padL} y1={y(t)} x2={padL + plotW} y2={y(t)} className="dd-chart-grid" />
                    <text x={padL - 8} y={y(t)} className="dd-chart-tick"
                          textAnchor="end" dominantBaseline="middle">{t}</text>
                </g>
            ))}
            <path d={line} className="dd-chart-line" fill="none" />
            {points.map((p, i) => {
                const py = y(Number(p.value)) + (hand ? jitter(i, 1.1) : 0);
                const isEst = (p.kind || '') === 'estimate';
                return (
                    <g key={i}>
                        <circle cx={x(i)} cy={py} r={3.4}
                                className={isEst ? 'dd-chart-dot-est' : 'dd-chart-dot'} />
                        <text x={x(i)} y={height - 10} className="dd-chart-tick"
                              textAnchor="middle">{txt(p.period)}</text>
                        {p.annotation && (
                            // Offset above the point, and flip to the left for
                            // the last two so end-of-series notes do not run off
                            // the canvas or sit on the line.
                            <text
                                x={i >= points.length - 2 ? x(i) - 8 : x(i)}
                                y={py - 12}
                                className="dd-chart-note"
                                textAnchor={i >= points.length - 2 ? 'end' : 'middle'}>
                                {txt(p.annotation)}
                            </text>
                        )}
                    </g>
                );
            })}
        </svg>
    );
};

/**
 * EPS x P/E sensitivity matrix.
 *
 * Rendered from the canonical numbers rather than a prose table so that no row
 * can go missing: the acceptance criteria require every EPS row to be visible,
 * and a hand-authored table is exactly where one quietly disappears.
 */
export const SensitivityMatrix = ({ epsRows, multiples, currentPrice }) => {
    const rows = arr(epsRows).filter(v => Number.isFinite(Number(v)));
    const cols = arr(multiples).filter(v => Number.isFinite(Number(v)));
    if (!rows.length || !cols.length) return null;
    return (
        <table className="dd-sens">
            <thead>
                <tr>
                    <th>EPS \ P/E</th>
                    {cols.map((m, i) => <th key={i}>{m}x</th>)}
                </tr>
            </thead>
            <tbody>
                {rows.map((eps, i) => (
                    <tr key={i}>
                        <th>${eps}</th>
                        {cols.map((m, j) => (
                            <td key={j}>${Math.round(eps * m).toLocaleString()}</td>
                        ))}
                    </tr>
                ))}
            </tbody>
            {currentPrice ? (
                <caption className="dd-sens-cap">
                    Illustrative share price = EPS × P/E. Current share price reference: {currentPrice}.
                </caption>
            ) : null}
        </table>
    );
};

/** Derives the sensitivity grid from the canonical object. */
const sensitivityFrom = (master) => {
    const fs = master?.financial_snapshot || {};
    const targets = arr(fs.management_targets);
    const epsTarget = targets.find(t => /eps/i.test(txt(t?.label)));
    const nums = txt(epsTarget?.value).match(/\d+(\.\d+)?/g) || [];
    let base = nums.length ? Number(nums[0]) : null;
    if (!base) {
        const m = txt(fs.eps).match(/\d+(\.\d+)?/g);
        base = m && m.length ? Number(m[m.length - 1]) : null;
    }
    if (!base || !Number.isFinite(base)) return null;
    const start = Math.max(5, Math.round(base / 5) * 5 - 5);
    return {
        epsRows: [start, start + 5, start + 10, start + 15],
        multiples: [17.5, 20, 22, 25],
    };
};

const Section = ({ n, title, note, className = '', children }) => (
    <section className={`dd-sec ${className}`}>
        <h3 className="dd-sec-h">
            {n ? <span className="dd-sec-n">{n}</span> : null}
            <span className="dd-sec-t">{title}</span>
            {note ? <span className="dd-sec-note">{note}</span> : null}
        </h3>
        {children}
    </section>
);

const SignpostTable = ({ signposts, dense = false }) => (
    <table className={`dd-signposts ${dense ? 'dd-signposts-dense' : ''}`}>
        <thead>
            <tr>
                <th>Signpost</th><th>Current</th><th>Target / Trigger</th><th>Why it matters</th>
            </tr>
        </thead>
        <tbody>
            {arr(signposts).map((s, i) => (
                <tr key={i}>
                    <th scope="row">{txt(s.signpost)}</th>
                    <td>{txt(s.current)}</td>
                    <td>{txt(s.target)}</td>
                    <td>{txt(s.why ?? s.why_it_matters)}</td>
                </tr>
            ))}
        </tbody>
    </table>
);

const BullBear = ({ bull, bear }) => (
    <div className="dd-bullbear">
        <div className="dd-bull">
            <div className="dd-bb-h">↗ Bull case</div>
            <ul>{arr(bull).map((b, i) => <li key={i}>{txt(b)}</li>)}</ul>
        </div>
        <div className="dd-vs">vs.</div>
        <div className="dd-bear">
            <div className="dd-bb-h">↓ Bear case</div>
            <ul>{arr(bear).map((b, i) => <li key={i}>{txt(b)}</li>)}</ul>
        </div>
    </div>
);

// ---------------------------------------------------------------------------
// 1 — One-Pager (Research Notebook, reference-calibrated)
// ---------------------------------------------------------------------------

export const DeepDiveOnePager = ({ data }) => {
    const d = data || {};
    const identity = d.identity || {};
    const segs = arr(d.segments);

    return (
        <div className="dd-page dd-onepager" data-dd-page="1">
            <header className="dd-op-head">
                <div className="dd-op-title">
                    <h1>{txt(d.company)} <span className="dd-tick">({txt(d.ticker)})</span></h1>
                    <p className="dd-op-sub">{txt(d.subheadline || d.headline)}</p>
                </div>
                <div className="dd-glance">
                    <div className="dd-glance-h">At a glance</div>
                    <ul>
                        {identity.exchange && <li>Ticker: {txt(d.ticker)} ({txt(identity.exchange)})</li>}
                        {identity.hq && <li>HQ: {txt(identity.hq)}</li>}
                        {identity.founded && <li>Founded: {txt(identity.founded)}</li>}
                        {identity.employees && <li>Employees: {txt(identity.employees)}</li>}
                        {identity.fy_end && <li>FY End: {txt(identity.fy_end)}</li>}
                        {identity.website && <li>{txt(identity.website)}</li>}
                    </ul>
                </div>
            </header>

            <div className="dd-op-grid">
                <Section n="1" title="Investment Thesis" className="dd-a">
                    <p className="dd-body">{txt(d.thesis_summary)}</p>
                    {d.core_question && (
                        <div className="dd-question">{txt(d.core_question)}</div>
                    )}
                    <ul className="dd-checks">
                        {arr(d.thesis_bullets).map((b, i) => <li key={i}>{txt(b)}</li>)}
                    </ul>
                </Section>

                <Section n="2" title="Company Overview" className="dd-b">
                    <p className="dd-body">{txt(d.overview_summary)}</p>
                    <div className="dd-seg-wrap">
                        <SegmentPie segments={segs} size={186} hand />
                        <ul className="dd-seg-list">
                            {segs.map((s, i) => (
                                <li key={i}>
                                    <span className="dd-seg-key"
                                          style={{ background: PIE_COLORS[i % PIE_COLORS.length] }} />
                                    <span>
                                        <b>{txt(s.short_name || s.name)}</b>
                                        {s.mix ? <em> {txt(s.mix)}</em> : null}
                                        <span className="dd-seg-d">{txt(s.description)}</span>
                                    </span>
                                </li>
                            ))}
                        </ul>
                    </div>
                    {d.other_profit_pool && (
                        <div className="dd-callout">{txt(d.other_profit_pool)}</div>
                    )}
                </Section>

                <Section n="3" title="Business Model" note="multiple profit pools" className="dd-c">
                    <div className="dd-bm">
                        {arr(d.business_model).map((b, i) => (
                            <div className="dd-bm-card" key={i}>
                                <div className="dd-bm-t">{txt(b.name)}</div>
                                <div className="dd-bm-d">{txt(b.description)}</div>
                            </div>
                        ))}
                    </div>
                    <div className="dd-bm-flow">Captures value across the customer life cycle →</div>
                </Section>

                <Section n="4" title="Key Opportunities" className="dd-d">
                    <ul className="dd-opps">
                        {arr(d.opportunities).map((o, i) => (
                            <li key={i}>
                                <b>{txt(o.title)}</b>
                                <span>{txt(o.detail)}</span>
                            </li>
                        ))}
                    </ul>
                </Section>

                <Section n="5" title="Financial Snapshot" className="dd-e">
                    <div className="dd-fin">
                        <ul className="dd-fin-bullets">
                            {arr(d.financial_bullets).map((b, i) => <li key={i}>{txt(b)}</li>)}
                        </ul>
                        <div className="dd-targets">
                            <div className="dd-targets-h">Mid-cycle targets</div>
                            {arr(d.targets).map((t, i) => (
                                <div className="dd-target" key={i}>
                                    <span className="dd-target-l">{txt(t.label)}</span>
                                    <span className="dd-target-v">{txt(t.value)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                    <EarningsChart history={d.earnings_history} width={545} height={150} hand />
                    <div className="dd-fin-foot">
                        {d.earnings_history?.cycle_note && (
                            <div className="dd-note">{txt(d.earnings_history.cycle_note)}</div>
                        )}
                        <div className="dd-val">
                            <div className="dd-val-h">Valuation (today)</div>
                            {arr(d.valuation_metrics).map((v, i) => (
                                <div className="dd-val-row" key={i}>
                                    <span>{txt(v.label)}</span><b>{txt(v.value)}</b>
                                </div>
                            ))}
                            {d.valuation_callout && (
                                <div className="dd-val-callout">{txt(d.valuation_callout)}</div>
                            )}
                        </div>
                    </div>
                </Section>

                <Section n="6" title="Key Signposts" note="what to watch" className="dd-f">
                    <SignpostTable signposts={d.signposts} dense />
                </Section>

                <Section n="7" title="Thesis Threats" note="what could break it" className="dd-g">
                    <ul className="dd-threats">
                        {arr(d.threats).map((t, i) => (
                            <li key={i}>
                                <b>{txt(t.threat)}</b>
                                <span>{txt(t.watch_for)}</span>
                            </li>
                        ))}
                    </ul>
                </Section>

                <Section title="★ Final Takeaway" className="dd-h">
                    <p className="dd-body">{txt(d.final_takeaway)}</p>
                    <BullBear bull={d.bull_case} bear={d.bear_case} />
                </Section>
            </div>

            <footer className="dd-op-foot">
                <span><b>Bottom line:</b> {txt(d.bottom_line)}</span>
                {d.secondary_bottom_line && (
                    <span className="dd-foot-tag">{txt(d.secondary_bottom_line)}</span>
                )}
            </footer>
        </div>
    );
};

// ---------------------------------------------------------------------------
// 2 — Two-Pager
// ---------------------------------------------------------------------------

const PageHead = ({ master, page, total, subtitle }) => {
    const g = master?.at_glance || {};
    return (
        <header className="dd-ph">
            <div>
                <div className="dd-ph-kicker">Investment Research · {txt(master?.ticker)} · Page {page}/{total}</div>
                <h1 className="dd-ph-title">
                    {txt(master?.company)} <span className="dd-tick">({txt(master?.ticker)})</span>
                </h1>
                <p className="dd-ph-sub">{subtitle}</p>
            </div>
            <div className="dd-ph-metrics">
                {g.share_price && <div><span>Price</span><b>{txt(g.share_price)}</b></div>}
                {g.market_cap && <div><span>Mkt cap</span><b>{txt(g.market_cap)}</b></div>}
                {master?.financial_snapshot?.forward_pe && (
                    <div><span>Fwd P/E</span><b>{txt(master.financial_snapshot.forward_pe)}</b></div>
                )}
            </div>
        </header>
    );
};

export const DeepDiveTwoPager = ({ master }) => {
    const m = master || {};
    const thesis = m.investment_thesis || {};
    const overview = m.company_overview || {};
    const fs = m.financial_snapshot || {};

    return (
        <>
            <div className="dd-page dd-twopager" data-dd-page="1">
                <PageHead master={m} page={1} total={2}
                          subtitle="Franchise, investment case and upside drivers" />

                <Section n="1" title="Investment Thesis">
                    <p className="dd-body dd-lg">{txt(thesis.summary)}</p>
                    {thesis.core_question && (
                        <div className="dd-question dd-lg">{txt(thesis.core_question)}</div>
                    )}
                    <ul className="dd-bullets dd-lg">
                        {arr(thesis.what_must_be_true).map((b, i) => <li key={i}>{txt(b)}</li>)}
                    </ul>
                </Section>

                <Section n="2" title="Company Overview">
                    <p className="dd-body dd-lg">{txt(overview.summary)}</p>
                    <div className="dd-seg-wrap dd-seg-wide">
                        <SegmentPie segments={overview.segments} size={230} />
                        <div className="dd-seg-rows">
                            {arr(overview.segments).map((s, i) => (
                                <div className="dd-seg-row" key={i}>
                                    <div className="dd-seg-row-t">
                                        <b>{txt(s.short_name || s.name)}</b>
                                        <span className="dd-seg-mix">{txt(s.mix)}</span>
                                    </div>
                                    <div className="dd-seg-d">{txt(s.description)}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                    {arr(overview.other_profit_pools).length > 0 && (
                        <div className="dd-callout">
                            {arr(overview.other_profit_pools).map(txt).join(' ')}
                        </div>
                    )}
                </Section>

                <div className="dd-two-col">
                    <Section n="3" title="Business Model">
                        <div className="dd-bm dd-bm-2">
                            {arr(m.business_model).map((b, i) => (
                                <div className="dd-bm-card" key={i}>
                                    <div className="dd-bm-t">{txt(b.name)}</div>
                                    <div className="dd-bm-d">{txt(b.description)}</div>
                                </div>
                            ))}
                        </div>
                    </Section>
                    <Section n="4" title="Key Opportunities">
                        <ul className="dd-opps dd-lg">
                            {arr(m.opportunities).map((o, i) => (
                                <li key={i}><b>{txt(o.title)}</b><span>{txt(o.detail)}</span></li>
                            ))}
                        </ul>
                    </Section>
                </div>

                <footer className="dd-pf">Page 1 · Franchise, investment case and upside drivers</footer>
            </div>

            <div className="dd-page dd-twopager" data-dd-page="2">
                <PageHead master={m} page={2} total={2}
                          subtitle="Earnings power, signposts and thesis-break conditions" />

                <Section n="5" title="Financial Snapshot">
                    <div className="dd-fin dd-fin-wide">
                        <ul className="dd-fin-bullets dd-lg">
                            {arr(fs.financial_bullets).map((b, i) => <li key={i}>{txt(b)}</li>)}
                        </ul>
                        <div className="dd-kpis">
                            {arr(fs.management_targets).map((t, i) => (
                                <div className="dd-kpi" key={i}>
                                    <span className="dd-kpi-l">{txt(t.label)}</span>
                                    <b className="dd-kpi-v">{txt(t.value)}</b>
                                    <span className="dd-kpi-c">{txt(t.context)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="dd-chart-row">
                        <div>
                            <div className="dd-chart-h">Earnings are cyclical</div>
                            <EarningsChart history={m.earnings_history} width={600} height={250} />
                            {m.earnings_history?.cycle_note && (
                                <div className="dd-note">{txt(m.earnings_history.cycle_note)}</div>
                            )}
                        </div>
                        <div className="dd-val dd-val-side">
                            <div className="dd-val-h">Valuation today</div>
                            <div className="dd-val-row"><span>Forward P/E</span><b>{txt(fs.forward_pe)}</b></div>
                            <div className="dd-val-row"><span>Historical</span><b>{txt(fs.historical_pe)}</b></div>
                            {fs.ev_ebitda && <div className="dd-val-row"><span>EV/EBITDA</span><b>{txt(fs.ev_ebitda)}</b></div>}
                            {fs.valuation_comment && (
                                <div className="dd-val-callout">{txt(fs.valuation_comment)}</div>
                            )}
                        </div>
                    </div>
                </Section>

                <Section n="6" title="Key Signposts">
                    <SignpostTable signposts={m.signposts} />
                </Section>

                <div className="dd-two-col">
                    <Section n="7" title="Thesis Threats">
                        <ul className="dd-threats dd-lg">
                            {arr(m.thesis_threats).map((t, i) => (
                                <li key={i}><b>{txt(t.threat)}</b><span>{txt(t.watch_for)}</span></li>
                            ))}
                        </ul>
                    </Section>
                    <Section title="★ Final Takeaway">
                        <p className="dd-body dd-lg">{txt(m.final_takeaway)}</p>
                        <BullBear bull={m.bull_case} bear={m.bear_case} />
                    </Section>
                </div>

                <footer className="dd-pf">
                    <b>Bottom line:</b> {txt(m.bottom_line)}
                </footer>
            </div>
        </>
    );
};

// ---------------------------------------------------------------------------
// 3 — Investment Memo
// ---------------------------------------------------------------------------

export const DeepDiveMemo = ({ master }) => {
    const m = master || {};
    const thesis = m.investment_thesis || {};
    const overview = m.company_overview || {};
    const fs = m.financial_snapshot || {};
    const sens = sensitivityFrom(m);
    const glance = m.at_glance || {};

    return (
        <>
            {/* Page 1 — business + thesis */}
            <div className="dd-page dd-memo" data-dd-page="1">
                <PageHead master={m} page={1} total={3}
                          subtitle="Franchise, investment thesis and what has to be true" />

                <Section n="1" title="Investment Thesis">
                    <p className="dd-body dd-md">{txt(thesis.summary)}</p>
                    {thesis.core_question && (
                        <div className="dd-question dd-md">{txt(thesis.core_question)}</div>
                    )}
                </Section>

                <div className="dd-three-col">
                    <Section title="What the market prices in">
                        <ul className="dd-bullets">
                            {arr(thesis.what_market_prices_in).map((b, i) => <li key={i}>{txt(b)}</li>)}
                        </ul>
                    </Section>
                    <Section title="What must be true">
                        <ul className="dd-bullets">
                            {arr(thesis.what_must_be_true).map((b, i) => <li key={i}>{txt(b)}</li>)}
                        </ul>
                    </Section>
                    <Section title="What would falsify it">
                        <ul className="dd-bullets dd-falsify">
                            {arr(thesis.falsification).map((b, i) => <li key={i}>{txt(b)}</li>)}
                        </ul>
                    </Section>
                </div>

                {thesis.variant_view && (
                    <div className="dd-variant">
                        <b>Variant view:</b> {txt(thesis.variant_view)}
                    </div>
                )}

                <Section n="2" title="Company Overview">
                    <p className="dd-body dd-md">{txt(overview.summary)}</p>
                    <div className="dd-seg-wrap dd-seg-wide">
                        <SegmentPie segments={overview.segments} size={200} />
                        <div className="dd-seg-rows">
                            {arr(overview.segments).map((s, i) => (
                                <div className="dd-seg-row" key={i}>
                                    <div className="dd-seg-row-t">
                                        <b>{txt(s.short_name || s.name)}</b>
                                        <span className="dd-seg-mix">{txt(s.mix)}</span>
                                    </div>
                                    <div className="dd-seg-d">{txt(s.description)}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </Section>

                <div className="dd-two-col">
                    <Section n="3" title="Business Model">
                        <div className="dd-bm dd-bm-2">
                            {arr(m.business_model).map((b, i) => (
                                <div className="dd-bm-card" key={i}>
                                    <div className="dd-bm-t">{txt(b.name)}</div>
                                    <div className="dd-bm-d">{txt(b.description)}</div>
                                </div>
                            ))}
                        </div>
                    </Section>
                    <Section n="4" title="Key Opportunities">
                        <ul className="dd-opps">
                            {arr(m.opportunities).map((o, i) => (
                                <li key={i}><b>{txt(o.title)}</b><span>{txt(o.detail)}</span></li>
                            ))}
                        </ul>
                    </Section>
                </div>

                <footer className="dd-pf">{txt(m.ticker)} · Equity Research<span>Page 1 / 3</span></footer>
            </div>

            {/* Page 2 — earnings, valuation, monitoring */}
            <div className="dd-page dd-memo" data-dd-page="2">
                <PageHead master={m} page={2} total={3}
                          subtitle="Earnings power, valuation and monitoring dashboard" />

                <Section n="5" title="Earnings Power & Valuation">
                    <div className="dd-kpi-grid">
                        {arr(fs.financial_bullets).slice(0, 6).map((b, i) => (
                            <div className="dd-kpi-card" key={i}>{txt(b)}</div>
                        ))}
                    </div>

                    <div className="dd-chart-row">
                        <div>
                            <div className="dd-chart-h">Earnings cycle</div>
                            <EarningsChart history={m.earnings_history} width={600} height={260} />
                            {m.earnings_history?.cycle_note && (
                                <div className="dd-note">{txt(m.earnings_history.cycle_note)}</div>
                            )}
                        </div>
                        <div>
                            <div className="dd-chart-h">Mid-cycle targets</div>
                            <div className="dd-kpis dd-kpis-2">
                                {arr(fs.management_targets).map((t, i) => (
                                    <div className="dd-kpi" key={i}>
                                        <span className="dd-kpi-l">{txt(t.label)}</span>
                                        <b className="dd-kpi-v">{txt(t.value)}</b>
                                        <span className="dd-kpi-c">{txt(t.context)}</span>
                                    </div>
                                ))}
                            </div>
                            <div className="dd-val">
                                <div className="dd-val-h">Valuation today</div>
                                <div className="dd-val-row"><span>Forward P/E</span><b>{txt(fs.forward_pe)}</b></div>
                                <div className="dd-val-row"><span>Historical P/E</span><b>{txt(fs.historical_pe)}</b></div>
                                {fs.valuation_comment && (
                                    <div className="dd-val-callout">{txt(fs.valuation_comment)}</div>
                                )}
                            </div>
                        </div>
                    </div>

                    {sens && (
                        <div className="dd-sens-wrap">
                            <div className="dd-chart-h">Mid-cycle EPS × P/E sensitivity</div>
                            <SensitivityMatrix epsRows={sens.epsRows} multiples={sens.multiples}
                                               currentPrice={txt(glance.share_price)} />
                        </div>
                    )}
                </Section>

                <Section n="6" title="Key Signposts — what to watch">
                    <SignpostTable signposts={m.signposts} />
                </Section>

                <footer className="dd-pf">{txt(m.ticker)} · Equity Research<span>Page 2 / 3</span></footer>
            </div>

            {/* Page 3 — decision framework */}
            <div className="dd-page dd-memo" data-dd-page="3">
                <PageHead master={m} page={3} total={3}
                          subtitle="Catalysts, thesis-break conditions and decision framework" />

                <Section n="7" title="Thesis Threats — explicit kill criteria">
                    <div className="dd-kill-grid">
                        {arr(m.thesis_threats).map((t, i) => (
                            <div className="dd-kill" key={i}>
                                <div className="dd-kill-t">{txt(t.threat)}</div>
                                <div className="dd-kill-d">{txt(t.watch_for)}</div>
                            </div>
                        ))}
                    </div>
                </Section>

                <Section n="8" title="Catalyst Calendar">
                    <div className="dd-cats">
                        {arr(m.catalysts).map((c, i) => (
                            <div className="dd-cat" key={i}>
                                <div className="dd-cat-when">{txt(c.timing)}</div>
                                <div className="dd-cat-what">{txt(c.event)}</div>
                                <div className="dd-cat-why">{txt(c.why_it_matters)}</div>
                            </div>
                        ))}
                    </div>
                </Section>

                <Section n="9" title="Scenario & Decision Framework">
                    <div className="dd-decision">
                        <table className="dd-scenarios">
                            <thead>
                                <tr><th>Scenario</th><th>Earnings</th><th>Multiple</th><th>Stock</th><th>What has to happen</th></tr>
                            </thead>
                            <tbody>
                                {arr(m.valuation_scenarios).map((s, i) => (
                                    <tr key={i} className={`dd-scn-${txt(s.case).toLowerCase()}`}>
                                        <th scope="row">{txt(s.case)}</th>
                                        <td>{txt(s.earnings)}</td>
                                        <td>{txt(s.multiple)}</td>
                                        <td>{txt(s.implied_value)}</td>
                                        <td>{txt(s.logic)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {/* Decision Lens deliberately replaces a separate evidence
                            checklist — the handoff calls out that duplicating the
                            same conditions in two blocks was a real regression. */}
                        <div className="dd-lens">
                            <div className="dd-lens-h">Decision lens</div>
                            <div className="dd-lens-block">
                                <div className="dd-lens-label">Get more constructive if</div>
                                <ul>{arr(thesis.what_must_be_true).map((b, i) => <li key={i}>{txt(b)}</li>)}</ul>
                            </div>
                            <div className="dd-lens-block dd-lens-neg">
                                <div className="dd-lens-label">Re-think the thesis if</div>
                                <ul>{arr(thesis.falsification).map((b, i) => <li key={i}>{txt(b)}</li>)}</ul>
                            </div>
                            {thesis.variant_view && (
                                <div className="dd-lens-variant">
                                    <b>Variant view:</b> {txt(thesis.variant_view)}
                                </div>
                            )}
                        </div>
                    </div>
                </Section>

                <Section title="Final Investment Takeaway">
                    <p className="dd-final">{txt(m.final_takeaway)}</p>
                    <div className="dd-note"><b>Bottom line:</b> {txt(m.bottom_line)}</div>
                </Section>

                <footer className="dd-pf">{txt(m.ticker)} · Equity Research<span>Page 3 / 3</span></footer>
            </div>
        </>
    );
};

// ---------------------------------------------------------------------------
// layout preflight
// ---------------------------------------------------------------------------

/**
 * Measures PAINTED child bounds, not scrollHeight.
 *
 * This is the single hardest-won lesson in the handoff. scrollHeight on a
 * stretched grid/flex container reports the stretched track height, which
 * produced false overflow errors that blocked visually perfect PDFs — while
 * genuinely clipped content inside a fixed-height child went undetected. So:
 * walk the real children and compare their painted rectangles against the page
 * box.
 */
export const preflightPages = (root) => {
    if (!root) return [];
    const pages = Array.from(root.querySelectorAll('[data-dd-page]'));
    return pages.map((page, idx) => {
        const box = page.getBoundingClientRect();
        const footer = page.querySelector('.dd-pf, .dd-op-foot');
        const issues = [];
        let maxBottom = 0, maxRight = 0;

        const walk = (el) => {
            for (const child of el.children) {
                const r = child.getBoundingClientRect();
                if (r.width === 0 && r.height === 0) continue;
                maxBottom = Math.max(maxBottom, r.bottom);
                maxRight = Math.max(maxRight, r.right);
                // A 1px tolerance absorbs sub-pixel rounding, which otherwise
                // reports a failure on a page that prints perfectly.
                if (r.bottom > box.bottom + 1) {
                    issues.push(`${child.className || child.tagName} overflows the page bottom by ${(r.bottom - box.bottom).toFixed(0)}px`);
                }
                if (r.right > box.right + 1) {
                    issues.push(`${child.className || child.tagName} overflows the page right edge by ${(r.right - box.right).toFixed(0)}px`);
                }
                walk(child);
            }
        };
        walk(page);

        if (footer) {
            const fr = footer.getBoundingClientRect();
            for (const child of page.children) {
                if (child === footer) continue;
                const r = child.getBoundingClientRect();
                if (r.bottom > fr.top + 1 && r.top < fr.bottom) {
                    issues.push(`${child.className || child.tagName} collides with the footer`);
                }
            }
        }

        // Content can overflow its GRID CELL without exceeding the page, which
        // is how the fifth opportunity and the last valuation row went missing
        // while the page-level check still said PASS. Each composed section is
        // its own box, so each one is measured against its own bounds.
        page.querySelectorAll('.dd-sec').forEach((sec) => {
            const sr = sec.getBoundingClientRect();
            for (const child of sec.children) {
                const r = child.getBoundingClientRect();
                if (r.width === 0 && r.height === 0) continue;
                if (r.bottom > sr.bottom + 1) {
                    const label = (sec.querySelector('.dd-sec-t') || {}).textContent || sec.className;
                    issues.push(`"${String(label).trim()}" has content cut off inside its box (${(r.bottom - sr.bottom).toFixed(0)}px)`);
                }
            }
        });

        // A table whose last row is cut is the failure mode that silently drops
        // a signpost or a sensitivity case, so it is checked by name.
        page.querySelectorAll('table').forEach((table) => {
            const rows = table.querySelectorAll('tbody tr');
            if (!rows.length) return;
            const last = rows[rows.length - 1].getBoundingClientRect();
            if (last.bottom > box.bottom + 1) {
                issues.push(`a table's final row is cut off (${rows.length} rows)`);
            }
        });

        // Utilization must measure FLOW content, not the footer. The footer is
        // absolutely positioned at the page bottom, so including it made every
        // page report exactly 100% — a number that looks like a measurement and
        // is actually a constant. The handoff sets real targets per artifact
        // (~90-96% one-pager, ~75-88% memo), which need a figure that can move.
        let flowBottom = box.top;
        for (const child of page.children) {
            if (child === footer) continue;
            const r = child.getBoundingClientRect();
            if (r.width === 0 && r.height === 0) continue;
            flowBottom = Math.max(flowBottom, r.bottom);
        }
        const used = box.height ? Math.min(1, (flowBottom - box.top) / box.height) : 0;
        return {
            page: idx + 1,
            ok: issues.length === 0,
            issues: [...new Set(issues)],
            utilization: Math.round(used * 100),
        };
    });
};
