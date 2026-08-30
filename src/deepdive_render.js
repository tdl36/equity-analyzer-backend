/* Deep Dive renderers — PORTED VERBATIM from Investment Research Studio v24.
 *
 * Source: charlie_investment_research_handoff/codebase/investment_research_studio_v24/static/app.js
 *
 * WHY THIS IS A COPY AND NOT A REWRITE
 * ---------------------------------------------------------------------------
 * The first attempt reimplemented these artifacts from the JSON schema and the
 * reviewed PDFs. It produced something structurally similar and visually
 * unrecognisable, because the calibration IS the product: the one-pager is an
 * SVG canvas with feTurbulence paper grain, hand-drawn nbFrame() boxes and HTML
 * dropped into foreignObject at absolute coordinates (X1=14, Y1=176, H1=390...),
 * with per-section accent colours, an icon set, product art and a logo. None of
 * that survives a paraphrase.
 *
 * So this file is the prototype's renderer, unchanged except for:
 *   - the trailing DOM wiring (lines 759+) which bootstrapped the Flask page;
 *   - `CURRENT` / `OUTPUT` becoming settable, since Charlie owns the data flow;
 *   - ES exports at the end.
 *
 * Do not "clean this up". Every magic number here was calibrated against
 * best_proven_outputs/DE_v29_*.pdf over dozens of iterations. Change it only
 * with a screenshot diff in front of you.
 *
 * The renderers emit HTML strings that Charlie injects. esc() escapes every
 * interpolated value at the point of use, which is what makes that safe.
 */


// --- Charlie shims ---------------------------------------------------------
// The prototype talked to a Flask page through these. The renderers never do,
// but sibling functions in the same file reference them, so they are stubbed
// rather than removed: deleting call sites would mean editing calibrated code.
// (the prototype's own `$` helper is defined below and is never called by the
// renderers; it only resolves at call time, so importing this module is safe
// outside a browser.)
let CURRENT = null;
let OUTPUT = 'report';

const $ = id => document.getElementById(id);
const esc = value => String(value ?? 'N/A').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const safeUrl = u => /^https?:\/\//i.test(String(u||'')) ? String(u) : '#';
const listHTML = (items, renderer=x=>esc(x)) => (items||[]).map(x=>`<li>${renderer(x)}</li>`).join('');

const TEMPLATE_NOTES = {
  twopager: {notebook: 'Readable notebook edition: same investment content across two pages with larger type.', institutional: 'Professional two-page investment brief with larger tables and charts.'},
  onepager: {
    notebook: 'Reference-calibrated notebook: heavier ink, larger readable type, denser tables and hero charts.',
    institutional: 'Professional buy-side brief: compact typography, high information density, clean hierarchy.',
    dashboard: 'Dark visual command center: KPI-forward, graphical, fast scanning.',
    strategy: 'Argument-first IC slide: priced-in expectations, variant view, scenarios and catalysts.',
    editorial: 'Premium finance-magazine composition: asymmetric, elegant, visual and selective.'
  },
  report: {
    institutional: 'Clean, professional, PM-friendly.',
    dashboard: 'More visual hierarchy and KPI-card emphasis.',
    strategy: 'Bolder, argument-led presentation with consulting-slide energy.'
  }
};

function setLoading(flag, msg='') {
  $('progress').classList.toggle('hidden', !flag);
  $('analyzeBtn').disabled = flag;
  $('reportView').classList.toggle('loading', flag);
  $('onepagerView').classList.toggle('loading', flag);
  if (msg) $('status').textContent = msg;
}

async function health() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    $('status').textContent = d.openai_enabled
      ? `Ready · AI + web research enabled · model ${d.model}`
      : 'Ready · OpenAI key not set. DE calibration + market-data mode available.';
  } catch (e) {
    $('status').textContent = 'Cannot reach the local backend. Restart the app and refresh.';
  }
}

async function analyze() {
  const ticker = $('ticker').value.trim().toUpperCase();
  if (!ticker) return;
  setLoading(true, `Researching ${ticker} → master report → dense one-page edit…`);
  const started = Date.now();
  try {
    const r = await fetch('/api/analyze', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ticker, force:$('forceRefresh').checked})
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Analysis failed');
    CURRENT = d;
    renderAll();
    const secs = Math.max(1, Math.round((Date.now()-started)/1000));
    const cached = d.meta?.cached ? ' · cache' : '';
    const violations = d.meta?.onepager_violations?.length || 0;
    $('status').textContent = `Done · ${d.master.company} (${d.master.ticker}) · ${secs}s${cached}${violations ? ` · ${violations} editorial warning(s)` : ''}`;
  } catch (e) {
    console.error(e);
    $('status').textContent = `Error: ${e.message}`;
  } finally { setLoading(false); }
}

async function loadDemo() {
  setLoading(true, 'Loading DE reference calibration…');
  try {
    const r = await fetch('/api/demo/de');
    CURRENT = await r.json();
    $('ticker').value = 'DE';
    renderAll();
    setOutput('onepager');
    $('onepagerTemplate').value = 'notebook';
    if($('twopagerTemplate')) $('twopagerTemplate').value='notebook';
    applyOnepagerTemplate();
    $('status').textContent = 'Loaded DE reference calibration · Research Notebook is the fidelity benchmark.';
  } catch (e) {
    $('status').textContent = `Demo error: ${e.message}`;
  } finally { setLoading(false); }
}

function setOutput(mode) {
  OUTPUT = mode;
  $('tabReport').classList.toggle('active', mode==='report');
  $('tabOnepager').classList.toggle('active', mode==='onepager');
  if($('tabTwopager')) $('tabTwopager').classList.toggle('active', mode==='twopager');
  $('reportView').classList.toggle('hidden', mode!=='report');
  $('onepagerView').classList.toggle('hidden', mode!=='onepager');
  if($('twopagerView')) $('twopagerView').classList.toggle('hidden', mode!=='twopager');
  $('reportControls').classList.toggle('hidden', mode!=='report');
  $('onepagerControls').classList.toggle('hidden', mode!=='onepager');
  if($('twopagerControls')) $('twopagerControls').classList.toggle('hidden', mode!=='twopager');
  $('currentOutputLabel').textContent = mode==='report' ? '3-Page Research Report' : mode==='onepager' ? 'Executive One-Pager · 1024 × 1536' : 'Readable Two-Pager · 2 × 1024 × 1536';
  if(mode==='twopager') renderTwopager();
  updateTopMeta();
  if (mode==='onepager') requestAnimationFrame(checkOnepagerFit);
}

function applyReportTemplate() {
  const t = $('reportTemplate').value;
  $('reportView').className = `view report-view theme-${t}` + (OUTPUT==='report' ? '' : ' hidden');
  $('reportTemplateNote').textContent = TEMPLATE_NOTES.report[t];
  updateTopMeta();
}

function applyOnepagerTemplate() {
  const t = $('onepagerTemplate').value;
  $('onepagerView').className = `view onepager-view template-${t}` + (OUTPUT==='onepager' ? '' : ' hidden');
  $('onepagerTemplateNote').textContent = TEMPLATE_NOTES.onepager[t];
  renderOnepager();
  updateTopMeta();
  requestAnimationFrame(checkOnepagerFit);
}

function applyTwopagerTemplate() {
  const t = $('twopagerTemplate').value;
  $('twopagerView').className = `view twopager-view twopager-${t}` + (OUTPUT==='twopager' ? '' : ' hidden');
  $('twopagerTemplateNote').textContent = TEMPLATE_NOTES.twopager[t];
  renderTwopager();
  updateTopMeta();
}

function toggleGrid(){
  $('onepagerView').classList.toggle('debug-grid', $('showGrid').checked);
}

function updateTopMeta() {
  if (!CURRENT) return $('topMeta').textContent = 'No analysis loaded';
  const m = CURRENT.master;
  if (OUTPUT==='report') $('topMeta').textContent = `${m.company} · ${$('reportTemplate').selectedOptions[0].textContent}`;
  else if (OUTPUT==='onepager') $('topMeta').textContent = `${m.company} · ${$('onepagerTemplate').selectedOptions[0].textContent}`;
  else $('topMeta').textContent = `${m.company} · ${$('twopagerTemplate').selectedOptions[0].textContent}`;
}

function websiteDomain(url) {
  try { return new URL(/^https?:\/\//.test(url)?url:`https://${url}`).hostname.replace(/^www\./,''); }
  catch { return ''; }
}
function logoHTML(d, cls='company-logo') {
  const website = CURRENT?.master?.at_glance?.website || d.identity?.website || '';
  const domain = websiteDomain(website);
  if (!domain) return `<div class="ticker-fallback">${esc(d.ticker)}</div>`;
  const src = `https://www.google.com/s2/favicons?domain_url=${encodeURIComponent('https://'+domain)}&sz=128`;
  return `<div class="${cls}"><img src="${src}" alt="${esc(d.company)} logo" onerror="this.style.display='none';this.nextElementSibling.style.display='grid'"><div class="ticker-fallback" style="display:none">${esc(d.ticker)}</div></div>`;
}

function reportHeader(m, page, descriptor='') {
  const g=m.at_glance||{};
  const pairs=[['Share Price',g.share_price],['Market Cap',g.market_cap],['Exchange',g.exchange],['Sector',g.sector],['Industry',g.industry],['As of',g.data_as_of]];
  return `<header class="report-header report-header-full"><div class="report-logo">${esc(m.ticker)}</div><div class="report-title"><div class="report-kicker">INVESTMENT RESEARCH · ${esc(m.ticker)} · PAGE ${page}/3</div><h1>${esc(m.company)} <span>(${esc(m.ticker)})</span></h1><p>${esc(descriptor||m.tagline)}</p></div><div class="report-glance">${pairs.map(x=>`<div><b>${esc(x[0])}</b><span>${esc(x[1])}</span></div>`).join('')}</div></header>`;
}
function reportRunningHeader(m,page,descriptor=''){
  return `<header class="report-running-header"><div><b>${esc(m.company)} <span>(${esc(m.ticker)})</span></b><small>${esc(descriptor)}</small></div><div>INVESTMENT RESEARCH · PAGE ${page}/3</div></header>`;
}
function reportFooter(m,p){return `<div class="report-footer"><span>${esc(m.ticker)} · Equity Research</span><span>Page ${p} / 3</span></div>`}
/* Share for a segment.
 *
 * mix_numeric is optional in practice -- models often return only the `mix`
 * label -- and when every segment lacked it the pie computed a zero total,
 * drew no wedges, and stacked every label at the same point. Cigna printed
 * "~17M%inimal": three labels on top of each other. Parse the label when the
 * numeric field is missing, and treat a non-numeric label ("Minimal") as zero. */
/* Segment mix by OPERATING PROFIT where the company discloses it.
 *
 * A revenue split tells you where the sales are; an equity investor needs to
 * know where the earnings are, and the two often disagree sharply. Eaton's
 * Mobility segment is 10% of revenue at roughly half the company margin, so a
 * revenue pie overstates what it contributes to the value of the business.
 * Profit share is used when it is available and the shares are coherent, and
 * the caption always says which basis is on screen -- an unlabelled pie that
 * silently changes meaning between companies would be worse than either.
 */
function hasProfitMix(segments){
  const segs = (segments || []).filter(Boolean);
  if (segs.length < 2) return false;
  const vals = segs.map(x => rawShare(x, 'profit'));
  if (vals.some(v => !(v > 0))) return false;
  const total = vals.reduce((a, b) => a + b, 0);
  return Math.abs(total - 100) <= 12;      // must actually add up
}
function rawShare(x, basis){
  const numeric = basis === 'profit'
    ? Number(x?.profit_mix_numeric)
    : Number(x?.mix_numeric ?? x?.value);
  if (Number.isFinite(numeric) && numeric > 0) return numeric;
  const label = basis === 'profit' ? x?.profit_mix : x?.mix;
  const m = String(label ?? '').match(/\d+(?:\.\d+)?/);
  return m ? Math.max(0, Number(m[0])) : 0;
}
/* Set once per render so the pie, the legend and the caption cannot disagree
   about which basis they are showing. */
let SEGMENT_BASIS = 'revenue';
function setSegmentBasis(segments){
  SEGMENT_BASIS = hasProfitMix(segments) ? 'profit' : 'revenue';
  return SEGMENT_BASIS;
}
function segmentShare(x){
  return rawShare(x, SEGMENT_BASIS);
}
function reportPieSVG(items, cls='report-pie-svg'){
  const vals=(items||[]).map(segmentShare), total=vals.reduce((a,b)=>a+b,0)||1;
  const cols=['#87ad58','#91b9dc','#e3bf55','#ad94c9']; let a=0;
  const paths=vals.map((v,i)=>{
    const a1=a+v/total*360,mid=(a+a1)/2;
    const p=`<path d="${pieArc(150,150,104,a,a1)}" fill="${cols[i%cols.length]}" stroke="#555" stroke-width="1.05"/>`;
    // A wedge too thin to hold its label gets none: the label cannot fit inside
    // the slice and lands on its neighbour's instead.
    let label='';
    if (v/total >= 0.06) {
      const pp=piePoint(150,150,67,mid);
      label=`<text x="${pp[0]}" y="${pp[1]}" text-anchor="middle" dominant-baseline="middle">${esc(items[i].mix||'')}</text>`;
    }
    a=a1; return p+label;
  }).join('');
  return `<svg class="${cls}" viewBox="0 0 300 300" preserveAspectRatio="xMidYMid meet">${paths}</svg>`;
}

function numsIn(value){
  return (String(value||'').match(/\d+(?:\.\d+)?/g)||[]).map(Number).filter(Number.isFinite);
}
function parseMoneyNumber(value){
  const n=numsIn(value)[0]; return Number.isFinite(n)?n:null;
}
/* Numbers that are calendar years are not financial quantities.
   `numsIn` harvests every number in a string, so an EPS written as
   "$8.50 (2027E)" -- the normal way a model writes it -- yielded [8.5, 2027].
   The matrix then treated 2027 as an earnings figure and produced rows of
   $2018.5 and $4045.5 EPS with $30,278 implied share prices for a $93 stock.
   The DE fixture writes "$40-45" with no year, which is why the golden master
   never caught it. */
function finNums(value){
  return (String(value||'')
    .replace(/\([^)]*\)/g, ' ')            // "(2027E)", "(±25bps)"
    .replace(/\b(?:19|20)\d{2}\b/g, ' ')   // bare years
    .match(/\d+(?:\.\d+)?/g) || []).map(Number).filter(Number.isFinite);
}
/* Pick n values spanning a sorted list, extending it when it is too short so
   the matrix always has a full set of rows/columns. */
/* Multiples, not every number in the sentence.
 *
 * historical_pe is prose: "10Y avg ~16x; current 27% below". Harvesting all of
 * it yielded 10 (from "10Y") and 27 (from "27%") as P/E multiples, which put a
 * 10x column half a turn from the 10.5x base case. Only a number actually
 * marked with an x counts; fall back to plain numbers when nothing is marked. */
function multipleNums(value){
  const s = String(value || '');
  const marked = [...s.matchAll(/(\d+(?:\.\d+)?)\s*[x\u00d7]/gi)].map(m => Number(m[1]));
  return marked.length ? marked : finNums(s);
}

function pickSpread(sorted, n){
  /* Collapse near-duplicates before choosing. Exact-match dedupe left Cigna
     with a "7.5x 10x 10.5x 16x" header: two columns half a turn apart carrying
     the same information while the span they were meant to cover went
     unrepresented. Values within a few percent of the range are one column. */
  const span=(sorted[sorted.length-1]-sorted[0])||sorted[0]||1;
  const tol=Math.max(span*0.06, 0.4);
  const u=[]; sorted.forEach(v=>{ if(!u.some(x=>Math.abs(x-v)<tol)) u.push(v); });
  if(u.length>=n){
    const out=[]; for(let i=0;i<n;i++) out.push(u[Math.round(i*(u.length-1)/(n-1))]);
    const ded=[]; out.forEach(v=>{ if(!ded.includes(v)) ded.push(v); });
    while(ded.length<n){ const gap=(ded[ded.length-1]-ded[0])/Math.max(1,ded.length-1)||ded[0]*.1; ded.push(Math.round((ded[ded.length-1]+gap)*100)/100); }
    return ded.slice(0,n);
  }
  if(!u.length) return [];
  const gaps=[]; for(let i=1;i<u.length;i++) gaps.push(u[i]-u[i-1]);
  const step=gaps.length ? gaps.reduce((a,b)=>a+b,0)/gaps.length : Math.max(u[0]*.12, .5);
  const out=u.slice();
  while(out.length<n) out.push(Math.round((out[out.length-1]+step)*100)/100);
  return out.slice(0,n);
}
/* Is this company's earnings history actually cyclical?
 *
 * The v24 template was calibrated on Deere, whose adjusted EPS swings from 45
 * to 16 and back, so it hard-codes "Earnings Are Cyclical" and "Mid-Cycle"
 * framing everywhere. On a secular compounder like Eaton -- earnings rising in
 * every year of the series -- those labels assert a business characteristic
 * that is not there, and "mid-cycle" EPS is a meaningless anchor when there is
 * no cycle to be mid of. Classify from the series instead of assuming.
 *
 * Cyclical means a real peak-to-trough drawdown AND more than one change of
 * direction; a single dip in an otherwise rising series is not a cycle.
 */
function earningsPattern(history) {
  const vals = (history?.points || [])
    .map(p => Number(p.value)).filter(Number.isFinite);
  if (vals.length < 4) return 'unknown';

  let peak = vals[0], maxDrawdown = 0;
  for (const v of vals) {
    if (v > peak) peak = v;
    if (peak > 0) maxDrawdown = Math.max(maxDrawdown, (peak - v) / peak);
  }
  let reversals = 0;
  for (let i = 2; i < vals.length; i++) {
    const a = Math.sign(vals[i - 1] - vals[i - 2]);
    const b = Math.sign(vals[i] - vals[i - 1]);
    if (a && b && a !== b) reversals++;
  }
  return (maxDrawdown >= 0.20 && reversals >= 2) ? 'cyclical' : 'secular';
}
/* Labels that follow from that, so a growth company is not described with a
   cycle vocabulary it does not have. */
function earningsLabels(history) {
  const cyclical = earningsPattern(history) === 'cyclical';
  return {
    cyclical,
    chartTitle:  cyclical ? 'Earnings Are Cyclical' : 'Earnings Trajectory',
    chartHead:   cyclical ? 'Earnings Cycle'        : 'Earnings Trajectory',
    matrixTitle: cyclical ? 'Mid-Cycle EPS \u00d7 P/E Sensitivity'
                          : 'Forward EPS \u00d7 P/E Sensitivity',
    targetsTitle: cyclical ? 'Mid-Cycle Targets'    : 'Management Targets',
    targetsTitleCaps: cyclical ? 'MID-CYCLE<br/>TARGETS' : 'MANAGEMENT<br/>TARGETS',
  };
}

/* Shorten a list for the one-pager only.
 *
 * The one-pager is a poster: its boxes are foreignObjects at fixed coordinates
 * and it is the densest of the three formats. The same bull/bear text that
 * reads well across two lines on the two-pager cannot fit five items there, and
 * with type size now a hard floor the box cannot absorb it by shrinking -- so
 * the last items were dropped instead. Shortening for this view keeps all five
 * points visible; the two-pager and memo still carry the full sentences from
 * the same source object.
 */
function posterList(items, maxChars) {
  return (items || []).map((raw) => {
    const t = String(raw || '').trim();
    if (t.length <= maxChars) return t;
    /* Stop at a clause, never mid-phrase. Cutting on a plain space produced
       "Data center backlog expands" from "...expands >228 GW" -- a bullet that
       reads as though the analyst trailed off. If there is no clause boundary
       to stop at, the item is left whole and the fitter scales the block. */
    const cut = t.slice(0, maxChars);
    for (const mark of ['; ', ', ', ' — ', ' – ']) {
      const i = cut.lastIndexOf(mark);
      if (i > maxChars * 0.55) return cut.slice(0, i).replace(/[,;:]$/, '');
    }
    return t;
  });
}

function reportSensitivityHTML(m,f){
  const scenarios=m.valuation_scenarios||[];
  const current=parseMoneyNumber(m.at_glance?.share_price);

  /* Rows come from the scenario EPS figures the memo already states, so the
     matrix agrees with the scenario table instead of being derived from an
     arithmetic step that could drift away from it. */
  let eps=[];
  scenarios.forEach(x=>{ finNums(x.earnings).forEach(n=>{ if(n>0 && n<1000) eps.push(n); }); });
  if(!eps.length) eps=finNums(f.eps).filter(n=>n>0 && n<1000);
  if(!eps.length) eps=[current?current/15:10];
  const rows=pickSpread(eps.slice().sort((a,b)=>a-b), 4)
    .map(v=>Math.round(v*100)/100);

  /* Columns: the scenario multiples plus the historical multiple, ascending.
     Previously these were [hist, ml, midpoint, mh] unsorted, which produced
     a "15x 12x 12.9x 14.2x" header -- out of order and effectively duplicated. */
  /* The bear/base/bull multiples are the ones the thesis actually argues over,
     so they take the columns first; the historical and forward multiples fill
     whatever is left. Treating all candidates equally let an even spread drop
     the base-case multiple entirely -- a sensitivity table whose own base case
     is missing. */
  const scenarioMults=[];
  scenarios.forEach(x=>{ multipleNums(x.multiple).forEach(n=>{ if(n>0 && n<100) scenarioMults.push(n); }); });
  const contextMults=[];
  multipleNums(f.historical_pe).forEach(n=>{ if(n>0 && n<100) contextMults.push(n); });
  multipleNums(f.forward_pe).forEach(n=>{ if(n>0 && n<100) contextMults.push(n); });
  let mults=scenarioMults.slice();
  contextMults.forEach(n=>{ if(mults.length<4 && !mults.some(x=>Math.abs(x-n)<Math.max(0.75, n*0.08))) mults.push(n); });
  if(mults.length<4) mults=mults.concat(contextMults);
  if(!mults.length) mults=[10,12,15,18];
  /* Reject multiples that are not part of the working range. Historical-P/E
     prose carries asides like "18-year avg ~20x; peak 37x at Sept 2024", and
     harvesting the 37 stretched the columns to 10x/17x/22x/37x -- a header
     dominated by a one-off peak rather than the range the thesis argues over.
     Anchor the band on the forward multiple when it is known, the median
     otherwise. */
  const anchorCands=finNums(f.forward_pe).filter(n=>n>0&&n<100);
  const sortedM=mults.slice().sort((a,b)=>a-b);
  const anchor=anchorCands.length ? anchorCands[0] : sortedM[Math.floor(sortedM.length/2)];
  const banded=sortedM.filter(n=>n>=anchor*0.5 && n<=anchor*2.2);
  const cols=pickSpread((banded.length>=2?banded:sortedM), 4)
    .map(v=>Math.round(v*10)/10);

  const fmtEps=e=>Number.isInteger(e)?String(e):e.toFixed(2).replace(/0$/,'');
  const cell=(e,mult)=>{const v=Math.round(e*mult); let c='neutral'; if(current){const r=v/current;c=r<.9?'down':r>1.1?'up':'near'} return `<td class="${c}"><b>$${v.toLocaleString()}</b></td>`};
  return `<div class="report-sensitivity"><div class="report-matrix-title">${earningsLabels(m.earnings_history).matrixTitle}</div><table><thead><tr><th>EPS \\ P/E</th>${cols.map(x=>`<th>${x}x</th>`).join('')}</tr></thead><tbody>${rows.map(e=>`<tr><th>$${fmtEps(e)}</th>${cols.map(mu=>cell(e,mu)).join('')}</tr>`).join('')}</tbody></table><small>Illustrative share price = EPS \u00d7 P/E. ${current?`Current share price reference: ${esc(String(m.at_glance?.share_price||'').startsWith('~')?m.at_glance.share_price:'~'+String(m.at_glance?.share_price||''))}.`:''}</small></div>`;
}
function reportSensitivityWideHTML(m,f){
  return `<div class="report-sensitivity-wide">${reportSensitivityHTML(m,f)}</div>`;
}
function reportDecisionLensHTML(t, includeVariant=true){
  const constructive=(t.what_must_be_true||[]).slice(0,3);
  const breaks=(t.falsification||[]).slice(0,3);
  return `<div class="report-decision-lens"><h3>Decision Lens</h3><div class="decision-lens-grid"><div class="constructive"><b>GET MORE CONSTRUCTIVE IF</b><ul>${listHTML(constructive)}</ul></div><div class="break"><b>RE-THINK THE THESIS IF</b><ul>${listHTML(breaks)}</ul></div></div>${includeVariant&&t.variant_view?`<p><b>Variant view:</b> ${esc(t.variant_view)}</p>`:''}</div>`;
}
function tpMarketSnapshot(m){
  const f=m.financial_snapshot||{}, g=m.at_glance||{};
  return `<div class="tp-market-snapshot"><div><span>PRICE</span><b>${esc(g.share_price)}</b></div><div><span>MKT CAP</span><b>${esc(g.market_cap)}</b></div><div><span>FWD P/E</span><b>${esc(f.forward_pe)}</b></div></div>`;
}
function tpSignpostTableHTML(items){
  const rows=(items||[]).slice(0,6).map(x=>`<tr><td><b>${esc(x.signpost)}</b></td><td>${nbRich(x.current)}</td><td>${nbRich(x.target)}</td><td>${esc(x.why)}</td></tr>`).join('');
  return `<table class="tp-signpost-table"><thead><tr><th>Signpost</th><th>Current</th><th>Target / Trigger</th><th>Why it matters</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function reportScenarioMatrixHTML(items){
  const rows=(items||[]).slice(0,3).map(x=>`<tr class="${String(x.case||'').toLowerCase()}"><th>${esc(x.case)}</th><td><b>${esc(x.earnings)}</b></td><td><b>${esc(x.multiple)}</b></td><td><b>${esc(x.implied_value)}</b></td><td>${esc(x.logic)}</td></tr>`).join('');
  return `<div class="report-scenario-matrix-wrap"><div class="report-matrix-title">Scenario Valuation Framework</div><table class="report-scenario-matrix"><thead><tr><th>Case</th><th>Earnings</th><th>Multiple</th><th>Stock outcome</th><th>Investment logic</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
function reportScenarioSideHTML(items){
  const rows=(items||[]).slice(0,3).map(x=>`<div class="report-side-scenario ${String(x.case||'').toLowerCase()}"><b>${esc(x.case)}</b><span>${esc(x.earnings)}</span><span>${esc(x.multiple)}</span><strong>${esc(x.implied_value)}</strong><small>${esc(x.logic)}</small></div>`).join('');
  return `<div class="report-side-scenarios"><h3>Scenario Framework</h3>${rows}</div>`;
}
function reportDecisionMatrixHTML(items){
  const rows=(items||[]).slice(0,3).map(x=>`<tr class="${String(x.case||'').toLowerCase()}"><th>${esc(x.case)}</th><td>${esc(x.earnings)}</td><td>${esc(x.multiple)}</td><td>${esc(x.implied_value)}</td><td>${esc(x.logic)}</td></tr>`).join('');
  return `<table class="report-decision-matrix"><thead><tr><th>Scenario</th><th>Earnings</th><th>Multiple</th><th>Stock implication</th><th>What has to happen</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderReport() {
  if (!CURRENT) return;
  const m=CURRENT.master, t=m.investment_thesis||{}, o=m.company_overview||{}, f=m.financial_snapshot||{};
  const pools=(m.business_model||[]).slice(0,4).map(x=>`<div class="report-pool"><b>${esc(x.name)}</b><span>${esc(x.description)}</span></div>`).join('');
  const segments=(o.segments||[]).slice(0,4).map(x=>`<div class="report-segment"><div><b>${esc(x.name)}</b><strong>${esc(x.mix)}</strong></div><p>${esc(x.description)}</p></div>`).join('');
  const opps=(m.opportunities||[]).slice(0,5).map(x=>`<div class="report-opp"><b>${esc(x.title)}</b><p>${esc(x.detail)}</p></div>`).join('');
  const primaryMetrics=[['Revenue',f.revenue,f.revenue_context],['Operating Margin',f.operating_margin,f.margin_context],['EPS',f.eps,f.eps_context],['Free Cash Flow',f.free_cash_flow,f.fcf_context],['Leverage',f.leverage,'Balance sheet'],['Returns',f.returns,'Capital efficiency']];
  const metrics=primaryMetrics.map(x=>`<div class="report-metric"><div class="value">${esc(x[1])}</div><div class="label">${esc(x[0])}</div><div class="context">${esc(x[2]||'')}</div></div>`).join('');
  const targets=(f.management_targets||[]).slice(0,4).map(x=>`<div class="report-target"><span>${esc(x.label)}</span><b>${esc(x.value)}</b><small>${esc(x.context||'')}</small></div>`).join('');
  const signs=(m.signposts||[]).slice(0,6).map(x=>`<tr><td><b>${esc(x.signpost)}</b></td><td>${esc(x.current)}</td><td>${esc(x.target)}</td><td>${esc(x.why_it_matters)}</td></tr>`).join('');
  const cats=(m.catalysts||[]).slice(0,4).map(x=>`<div class="report-catalyst"><b>${esc(x.timing)}</b><strong>${esc(x.event)}</strong><span>${esc(x.why_it_matters)}</span></div>`).join('');
  const threats=(m.thesis_threats||[]).slice(0,4).map(x=>`<div class="report-threat"><b>${esc(x.threat)}</b><p>${esc(x.watch_for)}</p></div>`).join('');
  const decisionMatrix=reportDecisionMatrixHTML(m.valuation_scenarios||[]);
  const reportChart=earningsChartSVG(m.earnings_history||CURRENT.onepager?.earnings_history||{},'report-earnings-chart');
  setSegmentBasis(o.segments);
  const companyPie=reportPieSVG((o.segments||[]).slice(0,4));
  const other=(o.other_profit_pools||[]).join(' · ');
  $('reportView').innerHTML = `
  <article class="report-page report-p1">${reportHeader(m,1,'Franchise, stock debate and business economics')}
    <main class="v21-report-p1">
      <section class="report-section v21-thesis strict-fit"><h2>1 · Investment Thesis</h2><p class="report-lead">${esc(t.summary)}</p><div class="report-question">${esc(t.core_question)}</div><div class="report-grid3 report-debate-grid"><div class="report-card"><div class="report-mini-title">What the market prices in</div><ul>${listHTML(t.what_market_prices_in)}</ul></div><div class="report-card"><div class="report-mini-title">What must be true</div><ul>${listHTML(t.what_must_be_true)}</ul></div><div class="report-card"><div class="report-mini-title">What would falsify it</div><ul>${listHTML(t.falsification)}</ul></div></div><div class="report-callout"><b>Variant view:</b> ${esc(t.variant_view)}</div></section>
      <section class="report-section v21-overview strict-fit"><h2>2 · Company Overview</h2><p>${esc(o.summary)}</p><div class="v21-overview-grid"><div class="report-pie-wrap">${companyPie}</div><div class="report-segments">${segments}${other?`<div class="v21-other"><b>Other profit pools:</b> ${esc(other)}</div>`:''}</div></div></section>
      <div class="v21-p1-bottom"><section class="report-section"><h2>3 · Business Model</h2><div class="report-pools">${pools}</div></section><section class="report-section"><h2>4 · Key Opportunities</h2><div class="report-opps">${opps}</div></section></div>
    </main>${reportFooter(m,1)}
  </article>
  <article class="report-page report-p2">${reportRunningHeader(m,2,'Earnings power, valuation and monitoring dashboard')}
    <main class="v21-report-p2">
      <section class="report-section v21-financial strict-fit"><h2>5 · Earnings Power & Valuation</h2><div class="report-metrics report-metrics-primary">${metrics}</div><div class="v21-fin-core"><div class="report-chart-wrap"><div class="report-chart-head"><h3>${earningsLabels(m.earnings_history).chartHead}</h3><span>${esc(m.earnings_history?.metric||'')}</span></div>${reportChart}<div class="report-cycle-note">${esc(m.earnings_history?.cycle_note||'')}</div></div><aside class="report-val-panel"><h3>${earningsLabels(m.earnings_history).targetsTitle}</h3><div class="report-target-grid">${targets}</div><div class="report-valuation-summary"><h3>Valuation Today</h3><div><span>Forward P/E</span><b>${esc(f.forward_pe)}</b></div><div><span>Historical P/E</span><b>${esc(f.historical_pe)}</b></div><p>${esc(f.valuation_comment)}</p></div></aside></div><div class="v21-sensitivity">${reportSensitivityWideHTML(m,f)}</div></section>
      <section class="report-section report-signposts v21-signposts strict-fit"><h2>6 · Key Signposts — What to Watch</h2><table class="report-table"><thead><tr><th>Signpost</th><th>Current</th><th>Target / Trigger</th><th>Why it matters</th></tr></thead><tbody>${signs}</tbody></table></section>
    </main>${reportFooter(m,2)}
  </article>
  <article class="report-page report-p3">${reportRunningHeader(m,3,'Catalysts, thesis-break conditions and decision framework')}
    <main class="v21-report-p3">
      <section class="report-section v21-threats strict-fit"><h2>7 · Thesis Threats — Explicit Kill Criteria</h2><div class="report-threat-grid">${threats}</div></section>
      <section class="report-section v21-catalysts strict-fit"><h2>8 · Catalyst Calendar</h2><div class="report-catalysts">${cats}</div></section>
      <section class="report-section v21-decision strict-fit"><h2>9 · Scenario & Decision Framework</h2><div class="v21-decision-grid">${decisionMatrix}${reportDecisionLensHTML(t,true)}</div></section>
      <section class="report-section v21-final strict-fit"><h2>Final Investment Takeaway</h2><div class="v21-final-copy">${esc(m.final_takeaway)}</div><div class="v21-bottom-line"><b>Bottom line:</b> ${esc(m.bottom_line)}</div></section>
    </main>${reportFooter(m,3)}
  </article>`;
}

// ---------- visual primitives ----------
const ICONS = {
  technology:'<svg viewBox="0 0 24 24"><path d="M4 13h3l2-8 4 14 2-6h5"/><circle cx="19" cy="5" r="2"/></svg>',
  growth:'<svg viewBox="0 0 24 24"><path d="M12 21V11M12 14C7 14 4 11 4 6c5 0 8 3 8 8ZM12 11c0-5 3-8 8-8 0 5-3 8-8 8Z"/></svg>',
  replacement:'<svg viewBox="0 0 24 24"><path d="M20 7v5h-5M4 17v-5h5M18 10a7 7 0 0 0-12-3L4 9M6 14a7 7 0 0 0 12 3l2-2"/></svg>',
  moat:'<svg viewBox="0 0 24 24"><path d="M5 21v-7h14v7M7 14V8h3v6M14 14V8h3v6M4 8h16L12 3 4 8Z"/></svg>',
  policy:'<svg viewBox="0 0 24 24"><path d="M3 21h18M5 18h14M6 18V9M10 18V9M14 18V9M18 18V9M4 9h16L12 3 4 9Z"/></svg>',
  product:'<svg viewBox="0 0 24 24"><path d="M4 7l8-4 8 4-8 4-8-4Zm0 0v11l8 4 8-4V7M12 11v11"/></svg>',
  margin:'<svg viewBox="0 0 24 24"><path d="M5 19 19 5M7 7h.01M17 17h.01"/><circle cx="7" cy="7" r="3"/><circle cx="17" cy="17" r="3"/></svg>',
  network:'<svg viewBox="0 0 24 24"><circle cx="5" cy="12" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="19" cy="18" r="2"/><path d="m7 11 10-4M7 13l10 4"/></svg>',
  data:'<svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/></svg>',
  capacity:'<svg viewBox="0 0 24 24"><path d="M4 19h16M6 16V8h12v8M9 8V5h6v3M8 12h8"/></svg>',
  cycle:'<svg viewBox="0 0 24 24"><path d="M20 7v5h-5M4 17v-5h5M18 10a7 7 0 0 0-12-3M6 14a7 7 0 0 0 12 3"/></svg>',
  adoption:'<svg viewBox="0 0 24 24"><path d="M4 20V11l8-6 8 6v11M8 20v-6h8v6"/><path d="m9 10 2 2 4-4"/></svg>',
  valuation:'<svg viewBox="0 0 24 24"><path d="M12 2v21M17 6.5c0-2-2-3-5-3s-5 1-5 3 2 3 5 3 5 1 5 3-2 3-5 3-5-1-5-3"/></svg>',
  regulation:'<svg viewBox="0 0 24 24"><path d="M12 3 4 6v6c0 5 3 8 8 9 5-1 8-4 8-9V6l-8-3Z"/><path d="m9 12 2 2 4-4"/></svg>',
  competition:'<svg viewBox="0 0 24 24"><path d="M7 4h10v5a5 5 0 0 1-10 0V5ZM9 21h6M12 14v7M7 7H4v2a4 4 0 0 0 4 4M17 7h3v2a4 4 0 0 1-4 4"/></svg>',
  execution:'<svg viewBox="0 0 24 24"><path d="M4 20h16M6 17l4-5 3 2 5-8"/><path d="m15 6 3 0 0 3"/></svg>'
};
function iconHTML(name){ return `<span class="line-icon">${ICONS[name]||ICONS.product}</span>`; }
function donutHTML(items, cls='') {
  const vals=items.map(x=>segmentShare(x)); const total=vals.reduce((a,b)=>a+b,0)||1;
  let cur=0; const colors=['var(--c1)','#83aee0','#e9c75c','#b89bd0']; const stops=[];
  vals.forEach((v,i)=>{let s=cur/total*100;cur+=v;let e=cur/total*100;stops.push(`${colors[i%colors.length]} ${s}% ${e}%`)});
  return `<div class="donut-wrap ${cls}"><div class="donut-chart" style="background:conic-gradient(${stops.join(',')})"><div></div></div><div class="donut-legend">${items.map((x,i)=>`<div><span style="background:${colors[i%colors.length]}"></span><b>${esc(x.short_name||x.label||x.name)}</b> ${esc(x.mix||x.value)}<small>${esc(x.description||x.detail||'')}</small></div>`).join('')}</div></div>`;
}
function flowHTML(items, cls='') { return `<div class="causal-flow ${cls}">${items.map((x,i)=>`${i?'<span class="flow-arrow">→</span>':''}<div class="flow-node"><b>${esc(x.label)}</b><small>${esc(x.detail)}</small></div>`).join('')}</div>`; }

/* Place chart annotations so they stay inside the plot and off each other.
 *
 * Every label used to be text-anchor="middle" at its own data point with no
 * bounds check and no collision handling. The first point sits on the y-axis,
 * so its label lost the half that fell at negative x -- "Pre-Aetna headwinds"
 * printed as "re-Aetna headwinds". Labels on the last points ran past the plot
 * and over the valuation panel beside it, and neighbouring labels simply
 * overprinted one another. */
/* The value at each point.
 *
 * The chart plotted a shape and labelled it with prose, so a reader could see
 * that earnings rose without being told what they were. The series is adjusted
 * EPS; printing the figure is the whole reason the chart exists. Values are
 * placed below the line where the free-text annotations sit above it, so the
 * two never compete for the same space.
 */
function valueLabelsSVG(pts, x, y, W, out) {
  const fmt = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return '';
    return n >= 100 ? `$${Math.round(n)}` : `$${n.toFixed(2).replace(/\.00$/, '')}`;
  };
  const placed = [];
  return pts.map((d, i) => {
    const text = fmt(d.value);
    if (!text) return '';
    const w = text.length * 4.9;
    let cx = x(i), anchor = 'middle';
    if (cx - w / 2 < 2) { anchor = 'start'; cx = 2; }
    else if (cx + w / 2 > W - 2) { anchor = 'end'; cx = W - 2; }
    const x0 = anchor === 'start' ? cx : anchor === 'end' ? cx - w : cx - w / 2;
    // Skip a label that would sit on the previous one rather than overprint.
    if (placed.some(p => x0 < p.x1 + 3 && x0 + w > p.x0 - 3)) return '';
    const yy = Math.min(y(Number(d.value)) + 15, 126);
    placed.push({ x0, x1: x0 + w, y: yy });
    if (out) out.push({ x0, x1: x0 + w, y: yy });
    return `<text x="${cx.toFixed(1)}" y="${yy.toFixed(1)}" text-anchor="${anchor}" `
         + `class="vlab${d.kind === 'estimate' ? ' est' : ''}">${esc(text)}</text>`;
  }).join('');
}

function annotationsSVG(pts, x, y, W, occupied) {
  const CHAR_W = 4.6;      // ~px per character at the 9px annotation size
  const LINE   = 10;
  const BOTTOM = 132;      // keep clear of the x-axis labels
  const MAX_CH = 34;       // a label wider than this cannot fit any placement
  const placed = Array.isArray(occupied) ? occupied.slice() : [];
  return pts.map((d, i) => ({ d, i })).filter(o => o.d.annotation).map(o => {
    /* Truncating these produced "Pandemic trough; portfolio restru..." across a
       shipped chart. An annotation is a caption: it is either readable or it is
       not there. The value label carries the number regardless. */
    const text = String(o.d.annotation);
    if (text.length > MAX_CH) return '';
    const w = text.length * CHAR_W;
    let cx = x(o.i), anchor = 'middle';
    if (cx - w / 2 < 2) { anchor = 'start'; cx = 2; }
    else if (cx + w / 2 > W - 2) { anchor = 'end'; cx = W - 2; }
    const x0 = anchor === 'start' ? cx : anchor === 'end' ? cx - w : cx - w / 2;
    const x1 = x0 + w;
    /* Try above the point first, then below. Stacking upward alone ran into
       the top of the plot and then gave up while still overlapping, which put
       two labels 2px apart -- unreadable. A label that cannot be placed
       anywhere clear is dropped: omitting one is honest, overprinting two
       destroys both. */
    const base = y(Number(o.d.value));
    const free = yy => yy >= 9 && yy <= BOTTOM
      && !placed.some(pl => !(x1 < pl.x0 - 2 || x0 > pl.x1 + 2)
                            && Math.abs(yy - pl.y) < LINE);
    let chosen = null;
    for (let k = 1; k <= 5 && chosen === null; k++) {
      const up = base - 10 - (k - 1) * LINE;
      if (free(up)) chosen = up;
    }
    for (let k = 1; k <= 4 && chosen === null; k++) {
      const down = base + 12 + (k - 1) * LINE;
      if (free(down)) chosen = down;
    }
    if (chosen === null) return '';
    placed.push({ x0, x1, y: chosen });
    if (Array.isArray(occupied)) occupied.push({ x0, x1, y: chosen });
    return `<text x="${cx.toFixed(1)}" y="${chosen.toFixed(1)}" text-anchor="${anchor}" class="anno">${esc(text)}</text>`;
  }).join('');
}

function earningsChartSVG(history, className = '') {
  const pts=(history?.points||[]).filter(x=>Number.isFinite(Number(x.value))).slice(-12);
  if (pts.length < 2) return `<div class="chart-empty">Historical series unavailable</div>`;
  const W=430,H=170,p={l:34,r:12,t:18,b:30};
  const vals=pts.map(x=>Number(x.value)); let min=Math.min(...vals,0),max=Math.max(...vals); if(max===min)max=min+1;
  const x=i=>p.l+i*(W-p.l-p.r)/(pts.length-1); const y=v=>p.t+(max-v)*(H-p.t-p.b)/(max-min);
  const actualIdx=pts.map((x,i)=>x.kind==='estimate'?i:null).filter(x=>x!==null)[0] ?? pts.length;
  const path=(arr,offset=0)=>arr.map((d,i)=>`${i?'L':'M'}${x(i+offset).toFixed(1)},${y(Number(d.value)).toFixed(1)}`).join(' ');
  const actual=pts.slice(0,Math.max(2,actualIdx)); const estimateStart=Math.max(0,actualIdx-1); const estimate=pts.slice(estimateStart);
  const ticks=[min,(min+max)/2,max];
  /* One occupancy list shared by both label passes. Placing values and
     annotations independently is what put "Consensus est." on top of the
     $15.20 label on a shipped chart: each pass believed its own space was
     free. */
  const _occupied=[];
  return `<svg class="earnings-chart ${className}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(history.metric||'Earnings')} history">
    ${ticks.map(v=>`<line x1="${p.l}" y1="${y(v)}" x2="${W-p.r}" y2="${y(v)}" class="gridline"/><text x="${p.l-6}" y="${y(v)+4}" text-anchor="end">${Math.round(v)}</text>`).join('')}
    <path d="${path(actual)}" class="series actual"/>
    ${estimate.length>1?`<path d="${path(estimate,estimateStart)}" class="series estimate"/>`:''}
    ${pts.map((d,i)=>`<circle cx="${x(i)}" cy="${y(Number(d.value))}" r="3" class="point ${d.kind==='estimate'?'est':''}"/>`).join('')}
    ${valueLabelsSVG(pts, x, y, W, _occupied)}
    ${pts.map((d,i)=>`<text x="${x(i)}" y="${H-10}" text-anchor="middle" class="xlab">${esc(d.period)}</text>`).join('')}
    ${annotationsSVG(pts, x, y, W, _occupied)}
  </svg>`;
}

function identityRows(d) {
  const x=d.identity||{};
  return [['Ticker',`${d.ticker} (${x.exchange||'N/A'})`],['HQ',x.hq],['Founded',x.founded],['Employees',x.employees],['FY End',x.fy_end],['Website',x.website]].map(r=>`<div><b>${esc(r[0])}:</b> ${esc(r[1])}</div>`).join('');
}
function sectionTitle(n,title,extra=''){return `<h2><span class="sec-num">${n}</span><span>${esc(title)}</span>${extra?`<small>${esc(extra)}</small>`:''}</h2>`}


function motifSVG(kind){
  const motifs={
    tractor:`<svg viewBox="0 0 150 90"><g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="42" cy="65" r="18"/><circle cx="112" cy="66" r="11"/><path d="M24 58h18l9-24h39l12 23h22v11h-9M55 34V21h25l12 14M64 20v14M82 41h19M28 58l9-17h18M55 55h44"/><path d="M85 16v-8M80 8h14"/></g></svg>`,
    excavator:`<svg viewBox="0 0 160 90"><g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M25 68h80l9 8H36zM44 68V43h34l13 25M52 43l9-16h25l13 18M84 26l26 2 24-20M132 8l13 13-17 19-15-8M108 30l20 10M20 75h104"/><circle cx="49" cy="74" r="6"/><circle cx="93" cy="74" r="6"/></g></svg>`,
    chip:`<svg viewBox="0 0 100 100"><g fill="none" stroke="currentColor" stroke-width="2"><rect x="25" y="25" width="50" height="50" rx="6"/><rect x="38" y="38" width="24" height="24"/><path d="M15 32h10M15 44h10M15 56h10M15 68h10M75 32h10M75 44h10M75 56h10M75 68h10M32 15v11M44 15v11M56 15v11M68 15v11M32 75v11M44 75v11M56 75v11M68 75v11"/></g></svg>`,
    bank:`<svg viewBox="0 0 110 90"><g fill="none" stroke="currentColor" stroke-width="2"><path d="M15 28h80L55 8 15 28ZM20 72h70M25 68V34M42 68V34M59 68V34M76 68V34M15 78h80"/></g></svg>`,
    vial:`<svg viewBox="0 0 90 100"><g fill="none" stroke="currentColor" stroke-width="2"><path d="M30 12h30M34 12v18L24 45v38h42V45L56 30V12M28 50h34M36 62h18M45 53v18"/></g></svg>`,
    oil:`<svg viewBox="0 0 100 100"><g fill="none" stroke="currentColor" stroke-width="2"><path d="M50 8 30 88M50 8l20 80M36 63h28M32 80h36M42 40h16M21 88h58M23 35h20M57 35h20M18 35l9-12h20M82 35l-9-12H53"/></g></svg>`,
    plane:`<svg viewBox="0 0 130 80"><g fill="none" stroke="currentColor" stroke-width="2"><path d="M8 43 57 37 82 8h13L82 36l37-2 8 8-45 6 16 21H86L60 50 18 57z"/></g></svg>`,
    cloud:`<svg viewBox="0 0 120 90"><g fill="none" stroke="currentColor" stroke-width="2"><path d="M27 67h64a18 18 0 0 0 1-36 30 30 0 0 0-56 8 14 14 0 0 0-9 28Z"/><path d="M45 51h30M52 43h16M52 59h16"/></g></svg>`,
    gear:`<svg viewBox="0 0 100 100"><g fill="none" stroke="currentColor" stroke-width="2"><circle cx="50" cy="50" r="17"/><path d="M50 12v13M50 75v13M12 50h13M75 50h13M23 23l9 9M68 68l9 9M77 23l-9 9M32 68l-9 9"/><circle cx="50" cy="50" r="31"/></g></svg>`
  }; return motifs[kind]||motifs.gear;
}
function companyMotifs(d,m){
  const t=String(d.ticker||'').toUpperCase(), sector=String(m?.at_glance?.sector||'').toLowerCase(), industry=String(m?.at_glance?.industry||'').toLowerCase();
  let kinds=['gear'];
  if(t==='DE' || industry.includes('agric') || industry.includes('machinery')) kinds=['tractor','excavator'];
  else if(industry.includes('semiconductor')) kinds=['chip','cloud'];
  else if(sector.includes('health') || industry.includes('biotech') || industry.includes('drug')) kinds=['vial','gear'];
  else if(sector.includes('financial') || industry.includes('bank')) kinds=['bank','cloud'];
  else if(sector.includes('energy') || industry.includes('oil')) kinds=['oil','gear'];
  else if(industry.includes('aerospace') || industry.includes('airline')) kinds=['plane','gear'];
  else if(sector.includes('technology') || industry.includes('software')) kinds=['cloud','chip'];
  return `<div class="company-motifs">${kinds.map((k,i)=>`<div class="motif motif-${i+1}">${motifSVG(k)}</div>`).join('')}</div>`;
}
function poolIconName(name=''){
  const s=name.toLowerCase();
  if(s.includes('finance')) return 'policy'; if(s.includes('service')||s.includes('after')) return 'replacement'; if(s.includes('software')||s.includes('technology')||s.includes('data')) return 'data'; return 'product';
}


// ---------- TEMPLATE 01: RESEARCH NOTEBOOK · SVG EDITION ----------
const NB_LOGOS = {
  DE:'/static/assets/de_logo_reference.png',
  NVDA:'https://cdn.simpleicons.org/nvidia/76B900',
  AAPL:'https://cdn.simpleicons.org/apple/111111',
  MSFT:'https://cdn.simpleicons.org/microsoft/5E5E5E',
  GOOGL:'https://cdn.simpleicons.org/google/4285F4',
  AMZN:'https://cdn.simpleicons.org/amazon/FF9900',
  META:'https://cdn.simpleicons.org/meta/0866FF',
  TSLA:'https://cdn.simpleicons.org/tesla/CC0000',
  JPM:'https://cdn.simpleicons.org/chase/117ACA',
  CAT:'https://cdn.simpleicons.org/caterpillar/FFCD11'
};
// Logo policy, following the handoff's rule: a real high-resolution mark when
// one is available, otherwise a crisp ticker plate. Deliberately NOT a favicon
// service -- those are 16-32px and upscaling them is called out explicitly as
// something never to do. Clearbit's logo API used to fill this gap but no longer
// resolves at all, and firing a request that always fails is worse than not
// firing one. So: curated marks, then the plate, which is a designed fallback
// rather than a failure state.
function nbLogoSrc(d){
  return NB_LOGOS[String(d.ticker||'').toUpperCase()] || '';
}

function nbLogoHTML(d){
  const src=nbLogoSrc(d);
  if(!src) return `<div class="nbv-logo-fallback">${esc(d.ticker)}</div>`;
  return `<div class="nbv-logo-wrap"><img src="${src}" alt="${esc(d.company)} logo" onerror="this.style.display='none';this.nextElementSibling.style.display='grid'"><div class="nbv-logo-fallback" style="display:none">${esc(d.ticker)}</div></div>`;
}
function fo(x,y,w,h,html,cls=''){
  return `<foreignObject x="${x}" y="${y}" width="${w}" height="${h}"><div xmlns="http://www.w3.org/1999/xhtml" class="nbv-fo ${cls}">${html}</div></foreignObject>`;
}
function nbFrame(x,y,w,h,cls=''){
  const j=((x+y+w+h)%7)-3, r=9, x2=x+w, y2=y+h;
  const d=`M ${x+r} ${y+.6*j} Q ${x+w*.35} ${y-.5*j} ${x2-r} ${y+.35*j} Q ${x2+1} ${y} ${x2+.3*j} ${y+r} L ${x2-.25*j} ${y2-r} Q ${x2} ${y2+1} ${x2-r} ${y2-.3*j} Q ${x+w*.55} ${y2+.5*j} ${x+r} ${y2-.25*j} Q ${x-1} ${y2} ${x+.2*j} ${y2-r} L ${x-.2*j} ${y+r} Q ${x} ${y} ${x+r} ${y+.6*j} Z`;
  return `<path d="${d}" class="nbv-frame ${cls}"/>`;
}
// Registrant names carry legal suffixes that no research note prints. Dropping
// them is what an analyst does by hand, and it is the difference between a
// title that fits and one that shrinks to unreadable or clips. Applied only
// when the full name will not fit, so short names are untouched.
const LEGAL_SUFFIX_RE = /,?\s+(incorporated|corporation|company limited|limited|holdings? (plc|inc|ltd)|plc|inc\.?|corp\.?|ltd\.?|s\.a\.?|n\.v\.?|ag|sa|nv)$/i;

function displayCompany(d){
  const full = String(d.company||'').trim();
  const ticker = String(d.ticker||'');
  if (full.length + ticker.length + 3 <= 34) return full;
  let short = full;
  // Strip repeatedly: "Taiwan Semiconductor Manufacturing Company Limited"
  // loses "Limited" then "Company".
  for (let i = 0; i < 3; i++) {
    const next = short.replace(LEGAL_SUFFIX_RE, '').trim();
    if (next === short || next.length < 4) break;
    short = next;
    if (short.length + ticker.length + 3 <= 34) break;
  }
  return short || full;
}

function nbTitleSize(d){
  // The floor used to be 31px, which still overflowed the header band for a
  // name like "UnitedHealth Group Incorporated" and printed as
  // "UNITEDHEALTH GROUP INCORPORAT". The ramp now continues down far enough
  // that any realistic registrant name fits on one line.
  // Measured against the header band, not guessed: the title is rendered in a
  // condensed display face at roughly 0.62em per character, and the band is
  // ~600px wide before the AT A GLANCE panel. "UnitedHealth Group Incorporated
  // (UNH)" is 37 characters and lost its final D at 29px, so the ramp is
  // derived rather than hand-tuned.
  const n=displayCompany(d).length + String(d.ticker||'').length + 3;
  const BAND_PX = 596, PER_CHAR_EM = 0.62;
  const fitted = Math.floor(BAND_PX / (n * PER_CHAR_EM));
  return Math.max(19, Math.min(41, fitted));
}
function nbSectionTitle(n,title,extra=''){
  return `<div class="nbv-title"><span class="nbv-num">${esc(n)}</span><b>${esc(title)}</b>${extra?`<small>${esc(extra)}</small>`:''}</div>`;
}
function piePoint(cx,cy,r,a){const rad=(a-90)*Math.PI/180;return [cx+r*Math.cos(rad),cy+r*Math.sin(rad)];}
function pieArc(cx,cy,r,a0,a1){
  const p0=piePoint(cx,cy,r,a0),p1=piePoint(cx,cy,r,a1),large=(a1-a0)>180?1:0;
  return `M ${cx} ${cy} L ${p0[0].toFixed(2)} ${p0[1].toFixed(2)} A ${r} ${r} 0 ${large} 1 ${p1[0].toFixed(2)} ${p1[1].toFixed(2)} Z`;
}
// A slice can only carry a couple of characters. `mix` is prose in the wild --
// UNH returns "72% rev, 55% op earnings" -- and printing that inside the wedge
// is what turned the chart into overlapping mush. Take the leading percentage
// when there is one, else the computed share, so the label always fits and
// always agrees with the geometry.
// The prototype hardcoded "(By Equipment Sales)", which is true of Deere and
// false of every other company. Say what the shares are actually of, and say
// nothing when the numbers do not add up rather than implying precision.
function segmentBasis(d){
  setSegmentBasis(d.segments);
  const segs = (d.segments||[]).filter(x=>segmentShare(x)>0);
  const total = segs.reduce((a,x)=>a+segmentShare(x),0);
  if (!segs.length) return '';
  const what = SEGMENT_BASIS === 'profit' ? 'segment operating profit' : 'revenue';
  return Math.abs(total-100) > 6
    ? `(share of ${what}, indicative)`
    : `(share of ${what})`;
}

function pieLabel(item, value, total){
  const share = total ? Math.round(value / total * 100) : 0;
  const stated_label = SEGMENT_BASIS === 'profit'
    ? ((item && item.profit_mix) || '') : ((item && item.mix) || '');
  const m = String(stated_label).match(/(\d{1,3}(?:\.\d)?)\s*%/);
  if (m) {
    const stated = Number(m[1]);
    // If the stated shares do not close to 100 the wedge cannot match the
    // label; show the normalised share so the picture is not a lie.
    if (Math.abs(total - 100) <= 6) return `${m[1]}%`;
    return `${share}%`;
  }
  return share ? `${share}%` : '';
}

function nbPieSVG(items,cx,cy,r){
  const vals=(items||[]).map(x=>segmentShare(x)), total=vals.reduce((a,b)=>a+b,0)||1;
  const cols=['#98bf65','#9fc3e7','#ebc85d','#baa2d2']; let a=0; const out=[];
  vals.forEach((v,i)=>{const a1=a+v/total*360;out.push(`<path class="nbv-pie-slice" d="${pieArc(cx,cy,r,a,a1)}" fill="${cols[i%cols.length]}" stroke="#3f3b34" stroke-width="1.05" filter="url(#nbRough)"/>`); const mid=(a+a1)/2, pt=piePoint(cx,cy,r*.58,mid); out.push(`<text x="${pt[0].toFixed(1)}" y="${pt[1].toFixed(1)}" class="nbv-pie-label" text-anchor="middle">${esc(pieLabel(items[i],v,total))}</text>`); a=a1;});
  return out.join('');
}
function nbIcon(name, size=24){return `<span class="nbv-icon" style="width:${size}px;height:${size}px">${ICONS[name]||ICONS.product}</span>`;}
const NUM_RE = /(\$?~?\d[\d,.]*(?:[-–]\d[\d,.]*)?%?x?\+?)/g;
// Highlight numbers on the RAW string, escaping each fragment as it is emitted.
// The original escaped first and highlighted second, so the number regex matched
// the "39" inside the &#39; that esc() had just produced for an apostrophe,
// split the entity across a <strong>, and the browser printed a literal
// "&#39;" -- which is why "Q3'24" rendered as "Q3&#39;24".
function nbRich(text){
  const raw = String(text ?? 'N/A');
  let out = '', last = 0, m;
  NUM_RE.lastIndex = 0;
  while ((m = NUM_RE.exec(raw)) !== null) {
    out += esc(raw.slice(last, m.index));
    out += `<strong class="nbv-numhi">${esc(m[0])}</strong>`;
    last = m.index + m[0].length;
  }
  return out + esc(raw.slice(last));
}
function nbIdentity(d){const x=d.identity||{};return [['Ticker',`${d.ticker} (${x.exchange||'N/A'})`],['HQ',x.hq],['Founded',x.founded],['Employees',x.employees],['FY End',x.fy_end],['Website',x.website]].map(r=>`<div><b>${esc(r[0])}:</b> ${esc(r[1])}</div>`).join('');}
function nbThesisArtwork(d){
  if(String(d.ticker||'').toUpperCase()!=='DE') return `<div class="nbv-thesis-note">${esc(d.subheadline||'')}</div>`;
  return `<div class="nbv-de-art"><img class="tractor" src="/static/assets/de_tractor_reference.png" alt="Deere tractor illustration"><img class="excavator" src="/static/assets/de_excavator_reference.png" alt="Deere excavator illustration"><div class="caption">More than machines.<br/>A data + lifecycle<br/>platform.</div></div>`;
}
function nbEarningsBridge(d){
  const vis=(d.visuals||[]).find(v=>v.type==='flow');
  const items=(vis?.items||[]).slice(0,4);
  if(!items.length) return '';
  return `<div class="nbv-bridge"><span class="bridge-kicker">THE EARNINGS UPGRADE</span>${items.map((x,i)=>`${i?'<i>→</i>':''}<div><b>${esc(x.label)}</b><small>${esc(x.detail)}</small></div>`).join('')}</div>`;
}


function notebookHTML(d,m){
  const X1=14,X2=520,W1=500,W2=490;
  const Y1=176,H1=390;
  const Y2=574,H2=218;
  const Y3=800,H3=438;
  const Y4=1246,H4=228;
  setSegmentBasis(d.segments);
  const segs=(d.segments||[]).slice(0,4);
  // The legend box is calibrated for three segments, which is what the DE
  // reference has. A four-segment company (UNH: UHC / Health / Rx / Insight)
  // overflowed it and the fourth row landed on top of the profit-pool note.
  // Tag the count so CSS can tighten the rows instead of losing one.
  const segCountCls = segs.length >= 4 ? ' nbv-seg-legend-4' : '';
  const segLegend=segs.map((x,i)=>`<div class="nbv-seg"><span class="swatch s${i}"></span><div class="nbv-segcopy"><b>${esc(x.short_name||x.name)}</b><strong>${esc(x.mix)}</strong><small>${esc(x.description)}</small></div></div>`).join('');
  const thesis=`${nbSectionTitle('1','INVESTMENT THESIS')}<p class="lead">${esc(d.thesis_summary)}</p><div class="nbv-question">${esc(d.core_question)}</div><ul class="nbv-checks">${(d.thesis_bullets||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>${nbThesisArtwork(d)}`;
  const overview=`${nbSectionTitle('2','COMPANY OVERVIEW')}<p class="lead">${esc(d.overview_summary)}</p><div class="nbv-minihead">KEY SEGMENTS <span>${esc(segmentBasis(d))}</span></div><div class="nbv-seg-legend${segCountCls}">${segLegend}</div><div class="nbv-profit-note">${esc(d.other_profit_pool)}</div>`;
  const business=`${nbSectionTitle('3','BUSINESS MODEL','— MULTIPLE PROFIT POOLS')}<div class="nbv-pools">${(d.business_model||[]).slice(0,4).map((x,i)=>`${i?'<i class="nbv-plus">+</i>':''}<div class="nbv-pool">${nbIcon(poolIconName(x.name),28)}<b>${esc(x.name)}</b><span>${esc(x.description)}</span></div>`).join('')}</div><div class="nbv-life">Captures value across the customer life cycle. <b>→</b></div>`;
  const opps=`${nbSectionTitle('4','KEY OPPORTUNITIES')}<div class="nbv-opps">${(d.opportunities||[]).slice(0,5).map(x=>`<div class="nbv-opp">${nbIcon(x.icon,31)}<div><b>${esc(x.title)}</b><span>${esc(x.detail)}</span></div></div>`).join('')}</div>`;
  const financial=`${nbSectionTitle('5','FINANCIAL SNAPSHOT','(FY / Latest)')}<div class="nbv-fin-top"><ul>${(d.financial_bullets||[]).slice(0,6).map(x=>`<li>${nbRich(x)}</li>`).join('')}</ul><div class="nbv-target"><h3>${earningsLabels(d.earnings_history).targetsTitleCaps}</h3>${(d.targets||[]).slice(0,4).map(x=>`<div><span>${esc(x.label)}</span><b>${esc(x.value)}</b></div>`).join('')}</div></div><div class="nbv-chart-title">${earningsLabels(d.earnings_history).chartTitle} <span>${esc(d.earnings_history?.metric||'')}</span></div><div class="nbv-fin-lower"><div class="nbv-chart">${earningsChartSVG(d.earnings_history,'nbv-chart-svg')}<div class="nbv-cycle-note">${esc(d.earnings_history?.cycle_note||'')}</div></div><div class="nbv-val"><h3>VALUATION <span>(Today)</span></h3>${(d.valuation_metrics||[]).slice(0,4).map(x=>`<div class="nbv-val-row"><span>${esc(x.label)}</span><b>${esc(x.value)}</b></div>`).join('')}<strong>${esc(d.valuation_callout)}</strong></div></div>`;
  const signs=`${nbSectionTitle('6','KEY SIGNPOSTS','(WHAT TO WATCH)')}<table class="nbv-table sign"><thead><tr><th>SIGNPOST</th><th>CURRENT</th><th>TARGET / TRIGGER</th><th>WHY IT MATTERS</th></tr></thead><tbody>${(d.signposts||[]).slice(0,6).map(x=>`<tr><td><b>${esc(x.signpost)}</b></td><td>${nbRich(x.current)}</td><td>${nbRich(x.target)}</td><td>${esc(x.why)}</td></tr>`).join('')}</tbody></table>`;
  const threats=`${nbSectionTitle('7','THESIS THREATS','(WHAT COULD BREAK IT)')}<table class="nbv-table threat"><tbody>${(d.threats||[]).slice(0,4).map(x=>`<tr><td class="ico">${nbIcon(x.icon,25)}</td><td><b>${esc(x.threat)}</b></td><td>${esc(x.watch_for)}</td></tr>`).join('')}</tbody></table>`;
  const final=`${nbSectionTitle('★','FINAL TAKEAWAY')}<p class="nbv-finalcopy">${esc(d.final_takeaway)}</p><div class="nbv-cases"><div class="case bull"><h3>↗ BULL CASE</h3><ul>${listHTML(posterList((d.bull_case||[]).slice(0,5), 34))}</ul></div><div class="vs">VS.</div><div class="case bear"><h3>↓ BEAR CASE</h3><ul>${listHTML(posterList((d.bear_case||[]).slice(0,5), 34))}</ul></div></div>`;
  return `<article class="op-canvas notebook-svg-canvas"><svg class="nbv-root" viewBox="0 0 1024 1536" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <filter id="paperNoise"><feTurbulence type="fractalNoise" baseFrequency=".82" numOctaves="2" seed="7" result="n"/><feColorMatrix in="n" type="matrix" values="0 0 0 0 0.25 0 0 0 0 0.23 0 0 0 0 0.18 0 0 0 .045 0"/></filter>
      <filter id="nbRough" x="-2%" y="-2%" width="104%" height="104%"><feTurbulence type="fractalNoise" baseFrequency="0.018 0.10" numOctaves="1" seed="9" result="noise"/><feDisplacementMap in="SourceGraphic" in2="noise" scale="0.55" xChannelSelector="R" yChannelSelector="B"/></filter>
    </defs>
    <rect width="1024" height="1536" fill="#f7f1e5"/><rect width="1024" height="1536" filter="url(#paperNoise)" opacity=".31"/>
    ${fo(18,12,196,154,nbLogoHTML(d),'nbv-logo-area')}
    ${fo(228,20,558,134,`<h1 style="font-size:${nbTitleSize(d)}px">${esc(displayCompany(d).toUpperCase())} <span>(${esc(d.ticker)})</span></h1><h2>${esc(d.headline)}</h2><div class="goldline"></div>`,'nbv-head')}
    ${nbFrame(804,14,206,154,'glance')}${fo(814,24,186,134,`<h3>AT A GLANCE</h3>${nbIdentity(d)}`,'nbv-glance')}
    <path d="M14 170 Q258 168 520 170 T1010 170" class="nbv-rule"/>
    ${nbFrame(X1,Y1,W1,H1)}${fo(X1+12,Y1+8,W1-24,H1-16,thesis,'nbv-section thesis')}
    ${nbFrame(X2,Y1,W2,H1)}${fo(X2+12,Y1+8,W2-24,H1-16,overview,'nbv-section overview')}
    ${nbPieSVG(segs,X2+113,Y1+236,101)}
    ${nbFrame(X1,Y2,W1,H2)}${fo(X1+12,Y2+8,W1-24,H2-16,business,'nbv-section business')}
    ${nbFrame(X2,Y2,W2,H2)}${fo(X2+12,Y2+8,W2-24,H2-16,opps,'nbv-section opps')}
    ${nbFrame(X1,Y3,W1,H3)}${fo(X1+12,Y3+8,W1-24,H3-16,financial,'nbv-section financial')}
    ${nbFrame(X2,Y3,W2,H3)}${fo(X2+12,Y3+8,W2-24,H3-16,signs,'nbv-section signs')}
    ${nbFrame(X1,Y4,W1,H4)}${fo(X1+12,Y4+8,W1-24,H4-16,threats,'nbv-section threats')}
    ${nbFrame(X2,Y4,W2,H4)}${fo(X2+12,Y4+8,W2-24,H4-16,final,'nbv-section final')}
    <path d="M14 1482 Q260 1480 520 1482 T1010 1482" class="nbv-rule"/><path d="M610 1507 q62 -5 116 0" class="nbv-pencil-line"/>
    ${fo(22,1488,980,38,`<div class="nbv-footer"><span>Bottom line: <b>${esc(d.bottom_line)}</b></span><strong>${esc(d.secondary_bottom_line)}</strong><i>◴</i></div>`,'')}
  </svg></article>`;
}


// ---------- READABLE TWO-PAGER · v13 REBUILD ----------
function tpHeader(d,page,subtitle=''){
  return `<header class="tp-header">${nbLogoHTML(d)}<div class="tp-header-title"><div class="tp-kicker">INVESTMENT RESEARCH · ${esc(d.ticker)} · PAGE ${page}/2</div><h1>${esc(displayCompany(d))} <span>(${esc(d.ticker)})</span></h1><p>${esc(subtitle||d.headline)}</p></div>${CURRENT?.master?tpMarketSnapshot(CURRENT.master):''}</header>`;
}
function tpRunningHeader(d,page,subtitle=''){
  return `<header class="tp-running-header"><div><b>${esc(displayCompany(d))} <span>(${esc(d.ticker)})</span></b><small>${esc(subtitle)}</small></div><em>PAGE ${page}/2</em></header>`;
}
function tpPieSVG(items){
  const vals=(items||[]).map(x=>segmentShare(x)), total=vals.reduce((a,b)=>a+b,0)||1;
  const cols=['#89b45b','#92bce0','#e6c255','#b69bd0']; let a=0;
  return `<svg class="tp-pie-svg" viewBox="0 0 300 300" preserveAspectRatio="xMidYMid meet">${vals.map((v,i)=>{const a1=a+v/total*360,mid=(a+a1)/2,pp=piePoint(150,150,68,mid);const out=`<path d="${pieArc(150,150,104,a,a1)}" fill="${cols[i%cols.length]}" stroke="#4a463f" stroke-width="1.15"/><text x="${pp[0]}" y="${pp[1]}" text-anchor="middle" dominant-baseline="middle">${esc(items[i].mix||'')}</text>`;a=a1;return out}).join('')}</svg>`;
}
function twopagerNotebookHTML(d,m){
  setSegmentBasis(d.segments);
  const segs=(d.segments||[]).slice(0,4);
  const p1=`<article class="tp-page tp-notebook-page tp-p1">${tpHeader(d,1,'Franchise, investment case and upside drivers')}<main class="t16-p1-grid">
    <section class="tp-thesis strict-fit"><h2>1 · Investment Thesis</h2><p class="tp-lead">${esc(d.thesis_summary)}</p><div class="tp-question">${esc(d.core_question)}</div><div class="tp-thesis-grid"><ul>${(d.thesis_bullets||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul><div class="tp-art">${nbThesisArtwork(d)}</div></div></section>
    <section class="tp-overview strict-fit"><h2>2 · Company Overview</h2><p>${esc(d.overview_summary)}</p><div class="t16-overview-grid"><div class="tp-pie-wrap">${tpPieSVG(segs)}</div><div class="tp-segments">${segs.map(x=>`<div><b>${esc(x.short_name||x.name)} <span>${esc(x.mix)}</span></b><p>${esc(x.description)}</p></div>`).join('')}<div class="t16-note"><b>Other profit pools:</b> ${esc(d.other_profit_pool)}</div></div></div></section>
    <div class="t16-lower"><section class="tp-business"><h2>3 · Business Model</h2><div class="t16-pools">${(d.business_model||[]).slice(0,4).map(x=>`<div>${nbIcon(poolIconName(x.name),34)}<b>${esc(x.name)}</b><span>${esc(x.description)}</span></div>`).join('')}</div><div class="tp-life">Captures value across the customer life cycle →</div></section><section class="tp-opportunities"><h2>4 · Key Opportunities</h2><div class="tp-opps">${(d.opportunities||[]).slice(0,5).map(x=>`<article>${nbIcon(x.icon,40)}<div><b>${esc(x.title)}</b><p>${esc(x.detail)}</p></div></article>`).join('')}</div></section></div>
  </main><footer>Page 1 · Franchise, investment case and upside drivers</footer></article>`;
  const p2=`<article class="tp-page tp-notebook-page tp-p2">${tpRunningHeader(d,2,'Earnings power, signposts and thesis-break conditions')}<main class="t16-p2-grid">
    <section class="tp-financial strict-fit"><h2>5 · Financial Snapshot</h2><div class="tp-fin-grid"><div class="tp-fin-bullets"><ul>${(d.financial_bullets||[]).slice(0,6).map(x=>`<li>${nbRich(x)}</li>`).join('')}</ul></div><div class="tp-targets">${(d.targets||[]).slice(0,4).map(x=>`<div><span>${esc(x.label)}</span><b>${esc(x.value)}</b><small>${esc(x.context||'')}</small></div>`).join('')}</div></div><div class="tp-chart-row"><div><h3>${earningsLabels(d.earnings_history).chartTitle}</h3>${earningsChartSVG(d.earnings_history,'tp-chart')}<p class="tp-cycle">${esc(d.earnings_history?.cycle_note||'')}</p></div><aside><h3>Valuation Today</h3>${(d.valuation_metrics||[]).slice(0,4).map(x=>`<div><span>${esc(x.label)}</span><b>${esc(x.value)}</b><small>${esc(x.context||'')}</small></div>`).join('')}<strong>${esc(d.valuation_callout)}</strong></aside></div></section>
    <section class="tp-signposts strict-fit"><h2>6 · Key Signposts</h2>${tpSignpostTableHTML(d.signposts||[])}</section>
    <section class="t16-bottom"><div class="tp-threats strict-fit"><h2>7 · Thesis Threats</h2>${(d.threats||[]).slice(0,4).map(x=>`<article>${nbIcon(x.icon,28)}<div><b>${esc(x.threat)}</b><p>${esc(x.watch_for)}</p></div></article>`).join('')}</div><div class="tp-final strict-fit"><h2>★ Final Takeaway</h2><p>${esc(d.final_takeaway)}</p><div class="tp-cases"><div><b>↗ BULL CASE</b><ul>${listHTML((d.bull_case||[]).slice(0,5))}</ul></div><i>VS.</i><div><b>↓ BEAR CASE</b><ul>${listHTML((d.bear_case||[]).slice(0,5))}</ul></div></div></div></section>
  </main><footer><b>Bottom line:</b> ${esc(d.bottom_line)} <span>${esc(d.secondary_bottom_line)}</span></footer></article>`;
  return p1+p2;
}
function twopagerInstitutionalHTML(d,m){return twopagerNotebookHTML(d,m).replaceAll('tp-notebook-page','tp-institutional-page');}
function renderTwopager(){
  if(!CURRENT||!$('twopagerView')) return;
  const d=CURRENT.onepager,m=CURRENT.master,t=$('twopagerTemplate').value;
  $('twopagerView').innerHTML=(t==='institutional'?twopagerInstitutionalHTML:twopagerNotebookHTML)(d,m);
  setTimeout(()=>updateLayoutQA(),0);
}

// ---------- TEMPLATE 02: INSTITUTIONAL ----------
function institutionalHTML(d,m){
  const f=m.financial_snapshot||{}, t=m.investment_thesis||{};
  const metrics=[['Price',m.at_glance?.share_price],['Mkt Cap',m.at_glance?.market_cap],['Fwd P/E',f.forward_pe],['Revenue',f.revenue],['Margin',f.operating_margin],['EPS',f.eps]];
  return `<article class="op-canvas inst-canvas">
    <header class="inst-header">${logoHTML(d,'inst-logo')}<div><div class="inst-kicker">EQUITY RESEARCH · ${esc(d.ticker)}</div><h1>${esc(d.company)}</h1><p>${esc(d.subheadline)}</p></div><div class="inst-metrics">${metrics.map(x=>`<div><span>${esc(x[0])}</span><b>${esc(x[1])}</b></div>`).join('')}</div></header>
    <section class="inst-thesis"><div class="inst-label">INVESTMENT THESIS</div><h2>${esc(d.core_question)}</h2><p>${esc(d.thesis_summary)}</p><div class="inst-thesis-grid"><div><b>PRICED IN</b><ul>${listHTML(t.what_market_prices_in)}</ul></div><div><b>MUST BE TRUE</b><ul>${listHTML(t.what_must_be_true)}</ul></div><div><b>FALSIFICATION</b><ul>${listHTML(t.falsification)}</ul></div></div></section>
    <section class="inst-overview"><div class="inst-label">BUSINESS & PROFIT POOLS</div><p>${esc(d.overview_summary)}</p><div class="inst-segment-row">${(d.segments||[]).map(x=>`<div><b>${esc(x.short_name||x.name)}</b><strong>${esc(x.mix)}</strong><span>${esc(x.description)}</span></div>`).join('')}</div><div class="inst-pools">${(d.business_model||[]).map(x=>`<div><b>${esc(x.name)}</b><span>${esc(x.description)}</span></div>`).join('')}</div></section>
    <section class="inst-financial"><div class="inst-label">EARNINGS, TARGETS & VALUATION</div><div class="inst-fin-grid"><div class="inst-fin-list">${(d.financial_bullets||[]).map(x=>`<div>• ${esc(x)}</div>`).join('')}</div><div class="inst-chart">${earningsChartSVG(d.earnings_history)}</div><div class="inst-targets">${(d.targets||[]).map(x=>`<div><span>${esc(x.label)}</span><b>${esc(x.value)}</b><small>${esc(x.context)}</small></div>`).join('')}</div></div><div class="inst-valuation">${(d.valuation_metrics||[]).map(x=>`<div><span>${esc(x.label)}</span><b>${esc(x.value)}</b></div>`).join('')}<strong>${esc(d.valuation_callout)}</strong></div></section>
    <section class="inst-opps"><div class="inst-label">UPSIDE DRIVERS</div>${(d.opportunities||[]).map(x=>`<div class="inst-opp">${iconHTML(x.icon)}<div><b>${esc(x.title)}</b><span>${esc(x.detail)}</span></div></div>`).join('')}</section>
    <section class="inst-signposts"><div class="inst-label">SIGNPOSTS</div><table><thead><tr><th>KPI</th><th>CURRENT</th><th>TRIGGER</th><th>WHY</th></tr></thead><tbody>${(d.signposts||[]).map(x=>`<tr><td>${esc(x.signpost)}</td><td>${nbRich(x.current)}</td><td>${nbRich(x.target)}</td><td>${esc(x.why)}</td></tr>`).join('')}</tbody></table></section>
    <section class="inst-risks"><div class="inst-label">THESIS THREATS</div>${(d.threats||[]).map(x=>`<div><b>${esc(x.threat)}</b><span>${esc(x.watch_for)}</span></div>`).join('')}</section>
    <section class="inst-cases"><div class="inst-label">SCENARIOS</div><div class="case bull"><b>BULL</b><ul>${listHTML((d.bull_case||[]).slice(0,5))}</ul></div><div class="case bear"><b>BEAR</b><ul>${listHTML((d.bear_case||[]).slice(0,5))}</ul></div></section>
    <footer class="inst-footer"><div><b>TAKEAWAY</b> ${esc(d.final_takeaway)}</div><strong>${esc(d.bottom_line)}</strong></footer>
  </article>`;
}

// ---------- TEMPLATE 03: DASHBOARD ----------
function dashboardHTML(d,m){
  const f=m.financial_snapshot||{}; const seg=(d.segments||[]);
  const kpis=[['PRICE',m.at_glance?.share_price],['MARKET CAP',m.at_glance?.market_cap],['FWD P/E',f.forward_pe],['REVENUE',f.revenue],['OP MARGIN',f.operating_margin],['EPS',f.eps]];
  return `<article class="op-canvas dash-canvas">
    <header class="dash-header"><div class="dash-id">${logoHTML(d,'dash-logo')}<div><div>${esc(d.ticker)} · ${esc(m.at_glance?.exchange)}</div><h1>${esc(d.company)}</h1><p>${esc(d.headline)}</p></div></div><div class="dash-kpis">${kpis.map(x=>`<div><span>${esc(x[0])}</span><b>${esc(x[1])}</b></div>`).join('')}</div></header>
    <section class="dash-debate"><div class="dash-label">CORE DEBATE</div><h2>${esc(d.core_question)}</h2><p>${esc(d.thesis_summary)}</p><div class="dash-bullets">${(d.thesis_bullets||[]).map(x=>`<span>${esc(x)}</span>`).join('')}</div></section>
    <section class="dash-mix"><div class="dash-label">BUSINESS MIX</div>${donutHTML(seg,'dash-donut')}<div class="dash-pools">${(d.business_model||[]).map(x=>`<div><b>${esc(x.name)}</b><span>${esc(x.description)}</span></div>`).join('')}</div></section>
    <section class="dash-chart"><div class="dash-label">EARNINGS / CYCLE</div>${earningsChartSVG(d.earnings_history,'dark-chart')}<div class="dash-chart-note">${esc(d.earnings_history?.cycle_note)}</div></section>
    <section class="dash-targets"><div class="dash-label">MANAGEMENT TARGETS</div>${(d.targets||[]).map(x=>`<div><span>${esc(x.label)}</span><b>${esc(x.value)}</b><small>${esc(x.context)}</small></div>`).join('')}<div class="dash-valuation">${(d.valuation_metrics||[]).map(x=>`<span><b>${esc(x.value)}</b>${esc(x.label)}</span>`).join('')}</div></section>
    <section class="dash-opps"><div class="dash-label">UPSIDE VECTORS</div>${(d.opportunities||[]).map((x,i)=>`<div>${iconHTML(x.icon)}<span><b>0${i+1}</b>${esc(x.title)}<small>${esc(x.detail)}</small></span></div>`).join('')}</section>
    <section class="dash-signals"><div class="dash-label">LIVE SIGNALS</div>${(d.signposts||[]).map(x=>`<div><b>${esc(x.signpost)}</b><span>${esc(x.current)}</span><i>→</i><strong>${esc(x.target)}</strong><small>${esc(x.why)}</small></div>`).join('')}</section>
    <section class="dash-risks"><div class="dash-label">RISK RADAR</div>${(d.threats||[]).map(x=>`<div>${iconHTML(x.icon)}<span><b>${esc(x.threat)}</b><small>${esc(x.watch_for)}</small></span></div>`).join('')}</section>
    <section class="dash-cases"><div class="dash-case bull"><span>BULL</span><ul>${listHTML((d.bull_case||[]).slice(0,5))}</ul></div><div class="dash-center">VS</div><div class="dash-case bear"><span>BEAR</span><ul>${listHTML((d.bear_case||[]).slice(0,5))}</ul></div></section>
    <footer class="dash-footer"><div><span>INVESTMENT TAKEAWAY</span>${esc(d.final_takeaway)}</div><strong>${esc(d.bottom_line)}</strong></footer>
  </article>`;
}

// ---------- TEMPLATE 04: STRATEGY ----------
function strategyHTML(d,m){
  const t=m.investment_thesis||{}; const vals=m.valuation_scenarios||[]; const cats=m.catalysts||[];
  return `<article class="op-canvas strategy-canvas">
    <header class="st-header"><div class="st-num">${esc(d.ticker)}</div><div><div class="st-kicker">INVESTMENT COMMITTEE · ONE PAGE</div><h1>${esc(d.company)}</h1><p>${esc(d.subheadline)}</p></div><div class="st-price"><span>PRICE</span><b>${esc(m.at_glance?.share_price)}</b><small>${esc(m.at_glance?.market_cap)} market cap</small></div></header>
    <section class="st-core"><div class="st-label">THE QUESTION</div><h2>${esc(d.core_question)}</h2><p>${esc(d.thesis_summary)}</p></section>
    <section class="st-pricing"><div class="st-label">WHAT THE MARKET PRICES IN</div><ol>${(t.what_market_prices_in||[]).map((x,i)=>`<li><b>0${i+1}</b>${esc(x)}</li>`).join('')}</ol></section>
    <section class="st-variant"><div class="st-label">VARIANT VIEW</div><blockquote>${esc(t.variant_view)}</blockquote><div class="st-must"><b>What must be true</b><ul>${listHTML(t.what_must_be_true)}</ul></div></section>
    <section class="st-bridge"><div class="st-label">EARNINGS BRIDGE</div>${flowHTML((d.visuals||[]).find(x=>x.type==='flow')?.items || [{label:'Cycle',detail:'Demand'},{label:'Mix',detail:'Content'},{label:'Margin',detail:'Economics'},{label:'EPS',detail:'Power'}],'strategy-flow')}<div class="st-targets">${(d.targets||[]).map(x=>`<div><span>${esc(x.label)}</span><b>${esc(x.value)}</b></div>`).join('')}</div></section>
    <section class="st-scenarios"><div class="st-label">VALUATION / SCENARIOS</div><div class="scenario-row">${vals.map(x=>`<div class="scenario ${String(x.case).toLowerCase()}"><span>${esc(x.case)}</span><b>${esc(x.earnings)}</b><strong>${esc(x.multiple)}</strong><small>${esc(x.logic)}</small></div>`).join('')}</div><div class="st-valcall">${esc(d.valuation_callout)}</div></section>
    <section class="st-signposts"><div class="st-label">6 THINGS THAT DECIDE THE STOCK</div>${(d.signposts||[]).map((x,i)=>`<div><b>${i+1}</b><span><strong>${esc(x.signpost)}</strong>${esc(x.current)}</span><i>→</i><span><strong>${esc(x.target)}</strong>${esc(x.why)}</span></div>`).join('')}</section>
    <section class="st-catalysts"><div class="st-label">CATALYST PATH</div>${cats.map(x=>`<div><b>${esc(x.timing)}</b><span>${esc(x.event)}</span><small>${esc(x.why_it_matters)}</small></div>`).join('')}</section>
    <section class="st-falsify"><div class="st-label">WHAT BREAKS THE THESIS</div><ul>${listHTML(t.falsification)}</ul>${(d.threats||[]).slice(0,2).map(x=>`<div><b>${esc(x.threat)}</b> · ${esc(x.watch_for)}</div>`).join('')}</section>
    <footer class="st-footer"><div>${esc(d.final_takeaway)}</div><strong>${esc(d.bottom_line)}</strong></footer>
  </article>`;
}

// ---------- TEMPLATE 05: EDITORIAL ----------
function editorialHTML(d,m){
  const g=m.at_glance||{}; const seg=(d.segments||[]);
  return `<article class="op-canvas editorial-canvas">
    <header class="ed-header"><div class="ed-brand">${logoHTML(d,'ed-logo')}<span>${esc(d.ticker)}</span></div><div class="ed-title"><div class="ed-kicker">THE INVESTMENT NOTE</div><h1>${esc(d.company)}</h1><h2>${esc(d.headline)}</h2></div><div class="ed-meta"><div>${esc(g.share_price)}<span>PRICE</span></div><div>${esc(g.market_cap)}<span>MARKET CAP</span></div><div>${esc(m.financial_snapshot?.forward_pe)}<span>FWD P/E</span></div></div></header>
    <section class="ed-hero"><blockquote>${esc(d.core_question)}</blockquote><p>${esc(d.thesis_summary)}</p><div class="ed-points">${(d.thesis_bullets||[]).map(x=>`<span>${esc(x)}</span>`).join('')}</div></section>
    <section class="ed-business"><div class="ed-label">THE BUSINESS</div><p>${esc(d.overview_summary)}</p>${donutHTML(seg,'ed-donut')}<div class="ed-pools">${(d.business_model||[]).map(x=>`<div><b>${esc(x.name)}</b><span>${esc(x.description)}</span></div>`).join('')}</div></section>
    <section class="ed-chart"><div class="ed-label">THE CYCLE</div><h3>${esc(d.earnings_history?.metric||'Earnings')} — history and the path implied by the thesis</h3>${earningsChartSVG(d.earnings_history,'editorial-chart')}<p>${esc(d.earnings_history?.cycle_note)}</p></section>
    <section class="ed-opps"><div class="ed-label">WHERE THE UPSIDE COMES FROM</div>${(d.opportunities||[]).map((x,i)=>`<article><span>0${i+1}</span><div><h3>${esc(x.title)}</h3><p>${esc(x.detail)}</p></div></article>`).join('')}</section>
    <section class="ed-numbers"><div class="ed-label">THE NUMBERS THAT MATTER</div><div class="ed-target-grid">${(d.targets||[]).map(x=>`<div><span>${esc(x.label)}</span><b>${esc(x.value)}</b><small>${esc(x.context)}</small></div>`).join('')}</div><div class="ed-val-grid">${(d.valuation_metrics||[]).map(x=>`<div><b>${esc(x.value)}</b><span>${esc(x.label)}</span></div>`).join('')}</div><strong>${esc(d.valuation_callout)}</strong></section>
    <section class="ed-watch"><div class="ed-label">WHAT TO WATCH</div>${(d.signposts||[]).map(x=>`<div><b>${esc(x.signpost)}</b><span>${esc(x.current)}</span><strong>${esc(x.target)}</strong></div>`).join('')}</section>
    <section class="ed-risk"><div class="ed-label">THE CONTRARY CASE</div>${(d.threats||[]).map(x=>`<div><b>${esc(x.threat)}</b><span>${esc(x.watch_for)}</span></div>`).join('')}</section>
    <footer class="ed-footer"><div><div class="ed-label">BOTTOM LINE</div><p>${esc(d.final_takeaway)}</p></div><blockquote>${esc(d.bottom_line)}</blockquote></footer>
  </article>`;
}

function renderOnepager(){
  if(!CURRENT) return;
  const d=CURRENT.onepager,m=CURRENT.master,t=$('onepagerTemplate').value;
  const renderers={notebook:notebookHTML,institutional:institutionalHTML,dashboard:dashboardHTML,strategy:strategyHTML,editorial:editorialHTML};
  $('onepagerView').innerHTML=(renderers[t]||notebookHTML)(d,m);
  requestAnimationFrame(checkOnepagerFit);
}

function fitNotebookLayout(){
  const canvas=$('onepagerView').querySelector('.notebook-svg-canvas');
  if(!canvas) return;
  canvas.dataset.layoutNeed='1536'; canvas.dataset.layoutAvailable='1536';
}

function checkOnepagerFit(){
  if($('onepagerTemplate').value==='notebook') fitNotebookLayout();
  const canvas=$('onepagerView').querySelector('.op-canvas');
  if(!canvas || !CURRENT || OUTPUT!=='onepager') return;
  const sections=[...canvas.querySelectorAll('.nb-box')]; const sectionOverflow=sections.some(el=>el.scrollHeight>el.clientHeight+2 || el.scrollWidth>el.clientWidth+2); const overflow = canvas.scrollHeight > canvas.clientHeight + 2 || canvas.scrollWidth > canvas.clientWidth + 2 || sectionOverflow;
  canvas.classList.toggle('overflow-warning',overflow);
  const apiWarnings=CURRENT.meta?.onepager_violations?.length||0;
  $('topMeta').textContent=`${CURRENT.master.company} · ${$('onepagerTemplate').selectedOptions[0].textContent}${overflow?' · LAYOUT OVERFLOW':''}${apiWarnings?` · ${apiWarnings} editorial warning(s)`:''}`;
}

function showSources(){
  if(!CURRENT) return alert('Run an analysis first.');
  const src=CURRENT.master?.sources||[];
  const w=window.open('','_blank','width=820,height=720,scrollbars=yes');
  if(!w) return alert('Please allow pop-ups to view the source trail.');
  const rows=src.length?src.map((s,i)=>`<li><a href="${safeUrl(s.url)}" target="_blank">${esc(s.title||s.url)}</a>${s.date?` <small>${esc(s.date)}</small>`:''}</li>`).join(''):'<p>No source list is stored in this calibration dataset.</p>';
  w.document.write(`<!doctype html><title>Research Source Trail</title><style>body{font:15px/1.5 Inter,Arial,sans-serif;max-width:760px;margin:40px auto;padding:0 24px;color:#1d2730}h1{font-size:26px}li{margin:10px 0}a{color:#164f7a}small{color:#777}</style><h1>${esc(CURRENT.master.company)} - Research Source Trail</h1><p>Sources are kept in the digital research record rather than occupying printed report space.</p><ol>${rows}</ol>`);
  w.document.close();
}


function measuredContentNeed(el){
  if(!el) return 0;
  const er=el.getBoundingClientRect();
  const cs=getComputedStyle(el);
  const pt=parseFloat(cs.paddingTop)||0, pb=parseFloat(cs.paddingBottom)||0;
  // Measure actual painted content, not the section's own stretched grid height.
  // Direct children are the authoritative layout units for our fixed-format sections.
  const kids=[...el.children].filter(n=>{
    const r=n.getBoundingClientRect();
    const s=getComputedStyle(n);
    return r.width>0 && r.height>0 && s.display!=='none' && s.visibility!=='hidden';
  });
  if(!kids.length) return Math.ceil(pt+pb+1);
  let maxBottom=er.top+pt;
  kids.forEach(n=>{
    const r=n.getBoundingClientRect();
    maxBottom=Math.max(maxBottom,r.bottom);
  });
  return Math.ceil(maxBottom-er.top+pb+2);
}
function rebalanceGridRows(grid, children, weights, safety=10){
  if(!grid || children.some(x=>!x)) return false;
  const cs=getComputedStyle(grid);
  const gap=parseFloat(cs.rowGap)||0;
  const pt=parseFloat(cs.paddingTop)||0, pb=parseFloat(cs.paddingBottom)||0;
  const available=grid.clientHeight-pt-pb-gap*(children.length-1);
  const desired=children.map(el=>Math.ceil(measuredContentNeed(el)+safety));
  const sum=desired.reduce((a,b)=>a+b,0);
  if(sum>available+2){
    // Content wants more room than the page has. The original bailed out here,
    // which left the fixed DE-calibrated rows in place -- so the rebalancer
    // helped only when things already fit, and did nothing in exactly the case
    // that produced clipping. Share the shortfall proportionally instead: every
    // section gets squeezed a little rather than the last ones losing their
    // content entirely. Content budgets are what stop it reaching this point;
    // this is about degrading gracefully when it does.
    const scale=available/sum;
    grid.style.gridTemplateRows=desired.map(d=>`${(d*scale).toFixed(1)}px`).join(' ');
    return true;
  }
  const extra=Math.max(0,available-sum);
  const ww=(weights&&weights.length===children.length)?weights:children.map(()=>1);
  const wsum=ww.reduce((a,b)=>a+b,0)||1;
  const rows=desired.map((d,i)=>d+extra*ww[i]/wsum);
  grid.style.gridTemplateRows=rows.map(x=>`${x.toFixed(1)}px`).join(' ');
  return true;
}
function rebalanceLongform(){
  // Re-measure the real painted content after the print class is active.
  const rp1=document.querySelector('.v21-report-p1');
  if(rp1) rebalanceGridRows(rp1,[rp1.querySelector('.v21-thesis'),rp1.querySelector('.v21-overview'),rp1.querySelector('.v21-p1-bottom')],[.32,.28,.40],12);
  const rp2=document.querySelector('.v21-report-p2');
  if(rp2) rebalanceGridRows(rp2,[rp2.querySelector('.v21-financial'),rp2.querySelector('.v21-signposts')],[.60,.40],12);
  const rp3=document.querySelector('.v21-report-p3');
  if(rp3) rebalanceGridRows(rp3,[rp3.querySelector('.v21-threats'),rp3.querySelector('.v21-catalysts'),rp3.querySelector('.v21-decision'),rp3.querySelector('.v21-final')],[.18,.12,.44,.26],12);
  const tp1=document.querySelector('.t16-p1-grid');
  if(tp1) rebalanceGridRows(tp1,[tp1.querySelector('.tp-thesis'),tp1.querySelector('.tp-overview'),tp1.querySelector('.t16-lower')],[.34,.27,.39],16);
  const tp2=document.querySelector('.t16-p2-grid');
  if(tp2) rebalanceGridRows(tp2,[tp2.querySelector('.tp-financial'),tp2.querySelector('.tp-signposts'),tp2.querySelector('.t16-bottom')],[.45,.31,.24],12);
}
function directContentOverflow(el,tol=6){
  if(!el) return false;
  const er=el.getBoundingClientRect();
  const kids=[...el.children].filter(n=>{
    const r=n.getBoundingClientRect(); const s=getComputedStyle(n);
    return r.width>0 && r.height>0 && s.display!=='none' && s.visibility!=='hidden';
  });
  return kids.some(n=>{const r=n.getBoundingClientRect();return r.bottom>er.bottom+tol || r.right>er.right+tol || r.left<er.left-tol;});
}
function strictClipFailures(){
  const failures=[];
  const visible=[...document.querySelectorAll('.strict-fit')].filter(el=>el.offsetParent!==null);
  visible.forEach(el=>{
    const er=el.getBoundingClientRect();
    const tol=6;
    // Do not use scrollHeight here: stretched grid/flex children create false positives on Safari/Chromium.
    // A failure means actual painted content crosses the section boundary.
    if(directContentOverflow(el,tol)){
      failures.push(`Clipped content: ${el.className.replace(/\\s+/g,' ')}`);
      return;
    }
    const candidates=[...el.querySelectorAll('tbody tr:last-child, .report-sensitivity small, .report-sensitivity tbody tr:last-child, article:last-child, .report-decision-lens, .v21-bottom-line')];
    candidates.forEach(n=>{
      const r=n.getBoundingClientRect();
      if(r.width>0 && r.height>0 && r.bottom>er.bottom+tol) failures.push(`Content cut off in ${el.className.replace(/\\s+/g,' ')}`);
    });
  });
  return [...new Set(failures)];
}

function collectLayoutQA(){
  const mode=OUTPUT;
  if(mode==='onepager'){
    const canvas=document.querySelector('.onepager-view:not(.hidden) .op-canvas');
    if(!canvas) return {failures:[],warnings:[]};
    const failures=canvas.classList.contains('overflow-warning')?['One-pager: layout overflow warning']:[];
    return {failures,warnings:[]};
  }
  const sel=mode==='report'?'.report-page':'.tp-page';
  const pages=[...document.querySelectorAll(sel)].filter(p=>p.offsetParent!==null);
  const failures=[...strictClipFailures()], warnings=[];
  pages.forEach((page,idx)=>{
    const pr=page.getBoundingClientRect();
    const footer=page.querySelector('.report-footer, footer');
    const fr=footer?footer.getBoundingClientRect():null;
    // Hard failures only: page-boundary escape or real footer collision.
    const protectedNodes=[...page.querySelectorAll('section, table, .report-pie-wrap, .tp-pie-wrap, .report-earnings-chart, .tp-chart-row')];
    protectedNodes.forEach(n=>{
      const r=n.getBoundingClientRect();
      if(r.width<1 || r.height<1) return;
      if(r.right>pr.right+5 || r.left<pr.left-5 || r.bottom>pr.bottom+5 || r.top<pr.top-5){
        failures.push(`Page ${idx+1}: content extends outside page boundary`);
      }
      if(fr && r.bottom>fr.top+3 && r.top<fr.top && !footer.contains(n)){
        failures.push(`Page ${idx+1}: content overlaps footer`);
      }
      // Only call something clipped if the element itself explicitly clips overflow.
      const cs=getComputedStyle(n);
      const clips=/hidden|clip/.test(cs.overflow+cs.overflowY+cs.overflowX);
      const isStrict=n.classList&&n.classList.contains('strict-fit');
      const realClip=isStrict?directContentOverflow(n,6):(n.scrollHeight>n.clientHeight+6 || n.scrollWidth>n.clientWidth+6);
      if(clips && realClip){
        failures.push(`Page ${idx+1}: clipped content in ${n.className||n.tagName.toLowerCase()}`);
      }
    });
    // Soft utilization + readability diagnostics only; these never block printing.
    const body=page.querySelector('.r16-p1-grid,.r16-p2-grid,.r16-p3-grid,.t16-p1-grid,.t16-p2-grid');
    if(body){
      const br=body.getBoundingClientRect();
      const content=[...body.children].filter(x=>x.getBoundingClientRect().height>0);
      if(content.length){
        const maxBottom=Math.max(...content.map(x=>x.getBoundingClientRect().bottom));
        const used=Math.max(0,Math.min(1,(maxBottom-br.top)/br.height));
        if(used<.80) warnings.push(`Page ${idx+1}: ${Math.round(used*100)}% vertical utilization`);
      }
      [...body.querySelectorAll('section')].forEach(sec=>{
        const sr=sec.getBoundingClientRect();
        const kids=[...sec.children].filter(x=>x.getBoundingClientRect().height>0);
        if(sr.height>180 && kids.length){
          const last=Math.max(...kids.map(x=>x.getBoundingClientRect().bottom));
          const blank=sr.bottom-last;
          if(blank>150) warnings.push(`Page ${idx+1}: ${Math.round(blank)}px unused in ${sec.className||'section'}`);
        }
      });
      const selector=mode==='report'?'.report-section p,.report-section li,.report-table td,.report-evidence span,.report-decision-matrix td':'.tp-page p,.tp-page li,.tp-signposts td,.tp-threats p,.tp-cases li';
      const sizes=[...page.querySelectorAll(selector)].filter(x=>x.textContent.trim()).map(x=>parseFloat(getComputedStyle(x).fontSize)).filter(Number.isFinite).sort((a,b)=>a-b);
      if(sizes.length){
        const med=sizes[Math.floor(sizes.length/2)], floor=mode==='report'?14:14.2;
        if(med<floor) warnings.push(`Page ${idx+1}: median substantive type ${med.toFixed(1)}px below ${floor}px target`);
      }
    }
  });
  return {failures:[...new Set(failures)],warnings:[...new Set(warnings)]};
}
function updateLayoutQA(){
  if(!CURRENT) return;
  const qa=collectLayoutQA();
  const suffix=qa.failures.length?` · QA: ${qa.failures.length} issue(s)` : qa.warnings.length?` · QA: ${qa.warnings.length} layout note(s)`:' · QA PASS';
  const base=OUTPUT==='report'?`${CURRENT.master.company} · ${$('reportTemplate').selectedOptions[0].textContent}`:OUTPUT==='twopager'?`${CURRENT.master.company} · ${$('twopagerTemplate').selectedOptions[0].textContent}`:`${CURRENT.master.company} · ${$('onepagerTemplate').selectedOptions[0].textContent}`;
  $('topMeta').textContent=base+suffix;
}

function renderAll(){ renderReport(); renderOnepager(); renderTwopager(); applyReportTemplate(); updateTopMeta(); setTimeout(()=>{rebalanceLongform();updateLayoutQA();},80); }
function exportJSON(){ if(!CURRENT)return alert('Run an analysis first.'); const blob=new Blob([JSON.stringify(CURRENT,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`${CURRENT.master.ticker}_research_v23.json`;a.click();URL.revokeObjectURL(a.href); }
async function printCurrent(){
  if(!CURRENT)return alert('Run an analysis first.');
  const cls=OUTPUT==='report'?'print-report':OUTPUT==='onepager'?'print-onepager':'print-twopager';
  document.body.classList.add(cls);
  await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
  rebalanceLongform();
  await new Promise(r=>requestAnimationFrame(r));
  const hard=strictClipFailures();
  const qa=collectLayoutQA();
  if(hard.length){
    document.body.classList.remove(cls);
    alert(`PDF export stopped because the layout would clip content. This is a hard invariant in v22.\n\n${hard.slice(0,8).join('\n')}\n\nPlease send the on-screen QA message rather than exporting a broken PDF.`);
    return;
  }
  if(qa.failures.length){
    document.body.classList.remove(cls);
    alert(`PDF export stopped because preflight found a page-boundary or footer collision:\n\n${qa.failures.slice(0,8).join('\n')}`);
    return;
  }
  window.print();
  setTimeout(()=>document.body.classList.remove('print-report','print-onepager','print-twopager'),300);
}


// --- exports ---------------------------------------------------------------
export function setCurrent(data) { CURRENT = data; }
export function setOutput_(kind) { OUTPUT = kind; }

export {
    notebookHTML, institutionalHTML, dashboardHTML, strategyHTML, editorialHTML,
    twopagerNotebookHTML, twopagerInstitutionalHTML,
    renderReport, earningsChartSVG, esc, listHTML,
    // The prototype's own layout QA. It was written against this exact DOM --
    // an SVG canvas whose HTML lives in foreignObject -- and it measures
    // painted need rather than naive rects, which is why it does not produce
    // the nonsense readings a generic walker gives on a scaled foreignObject.
    strictClipFailures, collectLayoutQA, rebalanceLongform, fitNotebookLayout,
};

/** The 3-page memo, as an HTML string.
 *
 * renderReport() writes into `#reportView` rather than returning markup, so it
 * is given a detached-but-attached mount and the result is read back out. That
 * keeps the calibrated memo renderer untouched instead of forking a second copy
 * that would drift from it.
 */
export function memoHTML(master, onepager) {
    if (typeof document === 'undefined') return '';
    const prior = document.getElementById('reportView');
    if (prior) prior.id = 'reportView__parked';
    const mount = document.createElement('div');
    mount.id = 'reportView';
    mount.style.cssText = 'position:absolute;left:-99999px;top:0;width:1024px;';
    document.body.appendChild(mount);
    try {
        CURRENT = { master, onepager };
        renderReport();
        return mount.innerHTML;
    } catch (e) {
        console.error('memoHTML failed:', e);
        return '';
    } finally {
        mount.remove();
        if (prior) prior.id = 'reportView';
    }
}
export { CURRENT, OUTPUT };
