import React, {useState, useEffect} from 'react';
import {parseDraft, canApprove, downloadFile, MAX_DRAFT_BYTES} from './thesis-import-model.mjs';

const labels = {pdfPages:'PDF pages', sourceId:'Source', sha256:'File hash', claimBasis:'Basis',
  thresholdBasis:'Threshold basis', reviewTrigger:'Review trigger', triggerPoints:'Trigger points',
  sourceRegister:'Source register', supportType:'Support', usageStatus:'Source-use note', reviewStatus:'Review coverage'};
const label = key => labels[key] || key.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase());
function Content({value}) {
  if (value == null || value === '') return <span className="ti-muted">Not supplied</span>;
  if (typeof value !== 'object') return <span style={{whiteSpace:'pre-wrap'}}>{String(value)}</span>;
  if (Array.isArray(value)) return <div className="ti-items">{value.length ? value.map((v,i) => <div key={i} className={typeof v === 'object' ? 'ti-item' : ''}><Content value={v}/></div>) : <span className="ti-muted">None</span>}</div>;
  return <dl>{Object.entries(value).filter(([k,v]) => k !== 'id' && v !== '' && v != null).map(([k,v]) => <div key={k} className="ti-field">
    {k === 'sources' ? <details><summary>Source references ({Array.isArray(v) ? v.length : 0})</summary><Content value={v}/></details> : <><dt>{label(k)}</dt><dd><Content value={v}/></dd></>}
  </div>)}</dl>;
}

export function ThesisImports({api='', initialTicker='', onNavigate, onSaved}) {
  const [ticker,setTicker] = useState(initialTicker);
  const [drafts,setDrafts] = useState([]), [draft,setDraft] = useState(null);
  const [busy,setBusy] = useState(false), [error,setError] = useState(''), [notice,setNotice] = useState('');
  const [checked,setChecked] = useState(false), [confirmation,setConfirmation] = useState('');
  const [loaded,setLoaded] = useState(false), [filter,setFilter] = useState('');
  async function call(path, body) {
    const r = await fetch(`${api}/api/thesis-imports${path}`, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || `Request failed (${r.status}). Your draft has not been discarded; refresh to check its status before retrying.`);
    return d;
  }
  async function list() { const data = await call(filter ? `?ticker=${encodeURIComponent(filter)}` : ''); setDrafts(data.drafts); setLoaded(true); }
  async function select(id) { const d = await call(`/${id}`); setDraft(d); setTicker(d.ticker); setChecked(false); setConfirmation(''); }
  async function act(fn) { setBusy(true); setError(''); setNotice(''); try { await fn(); } catch(e) { setError(e.message); } finally { setBusy(false); } }
  useEffect(() => { let active=true; call('').then(d=>{if(active){setDrafts(d.drafts);setLoaded(true);}}).catch(e=>{if(active)setError(e.message);}); return ()=>{active=false;}; }, [api]);
  async function prepare() {
    if (!/^[A-Z0-9][A-Z0-9.-]{0,19}$/.test(ticker)) throw new Error('Enter a ticker before preparing a thesis.');
    const data = await call(`/prepare/${encodeURIComponent(ticker)}`);
    downloadFile(`${ticker}-prepare-for-ChatGPT.json`, {...data.package, instructions:data.instructions});
    setNotice('Preparation file downloaded. Attach it and your chosen source documents in ChatGPT. Ask for the completed JSON draft, then import that file here.');
  }
  async function upload(file) {
    if (!file) return;
    if (file.size > MAX_DRAFT_BYTES) throw new Error('Choose a draft smaller than 2 MB.');
    const p = parseDraft(await file.text());
    const d = await call('', p); await select(d.id); await list();
    setNotice(d.replayed ? 'This file is already in your inbox. Its saved review is open below.' : 'Draft saved for review. The live thesis has not changed.');
  }
  async function approve() {
    await call(`/${draft.id}/approve`, {confirm:true,ticker:confirmation,fingerprint:draft.fingerprint});
    await select(draft.id); await list();
    setNotice('Thesis approved and saved, with source references and a restorable revision. No model calls were made.');
    onSaved?.();
  }
  return <div className="ti-workspace">
    <style>{`
      .ti-workspace{overflow:auto;padding:24px 24px 100px;flex:1;min-width:0;color:#000;background:#f4f3ef;font:16px/1.5 Calibri,Carlito,sans-serif}
      .ti-workspace *{box-sizing:border-box;font-family:Calibri,Carlito,sans-serif!important;color:#000!important}.ti-workspace input::placeholder{color:#000;opacity:.65}.ti-inner{max-width:1260px;margin:auto}.ti-workspace h1{font-size:30px;font-weight:700;margin:0}.ti-workspace h2{font-size:21px;font-weight:700;margin:0 0 10px}.ti-workspace h3{font-weight:700}
      .ti-workspace p{margin:8px 0}.ti-muted{font-size:14px}.ti-kicker{font-size:12px;letter-spacing:.13em;text-transform:uppercase}.ti-actions{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:16px 0}.ti-workspace button,.ti-upload{border:1px solid #aaa;border-radius:7px;padding:9px 14px;background:#fff;color:#000;cursor:pointer;font-weight:600}
      .ti-workspace button:disabled{opacity:.5;cursor:wait}.ti-workspace button:focus-visible,.ti-workspace input:focus-visible,.ti-upload:focus-within{outline:3px solid #647c8a;outline-offset:2px}.ti-workspace input[type=text]{min-width:0;border:1px solid #999;border-radius:5px;padding:9px;background:white;color:#000}.ti-workspace .ti-primary{background:#dbe8d5;border-color:#7a8c70}.ti-upload input{position:absolute;opacity:0;width:1px;height:1px}.ti-panel{padding:22px;background:#fff;border:1px solid #d8d5cc;border-radius:10px;margin-top:20px}.ti-message{padding:13px 16px;border:1px solid #b0baaa;background:#eaf1e5;border-radius:7px;margin:15px 0;overflow-wrap:anywhere}.ti-error{background:#fbe7df;border-color:#c28d77}
      .ti-inbox{display:flex;gap:8px;overflow:auto;padding:5px 0 10px}.ti-inbox button{text-align:left;flex-shrink:0;max-width:270px;white-space:normal}.ti-inbox button[aria-pressed=true]{border:2px solid #485744;background:#eaf1e5}.ti-status{font-size:12px;text-transform:uppercase;letter-spacing:.06em}.ti-section{border:1px solid #d8d5cc;border-radius:8px;overflow:hidden;margin:18px 0}.ti-section-head{padding:12px 16px;background:#eeece5;display:flex;justify-content:space-between;gap:12px}.ti-comparison{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.ti-column{padding:18px;min-width:0;overflow-wrap:anywhere}.ti-column+.ti-column{border-left:1px solid #ddd}.ti-column-title{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:14px}
      .ti-field{margin:8px 0}.ti-field dt{font-size:13px;font-weight:700}.ti-field dd{margin:2px 0 10px}.ti-item{border-top:1px solid #ddd;padding:10px 0}.ti-workspace details{margin:12px 0;overflow-wrap:anywhere}.ti-workspace summary{cursor:pointer;font-weight:600}.ti-workspace dl{margin:0}.ti-approval{background:#f1f4ed}.ti-check{display:flex;align-items:flex-start;gap:10px}.ti-check input{margin-top:6px;flex-shrink:0}.ti-workspace ul{list-style:disc;padding-left:22px}.ti-guide{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;margin:22px 0}.ti-guide strong{display:block}.ti-busy{font-size:14px}
      @media(max-width:700px){.ti-workspace{padding:16px 12px 110px}.ti-guide{grid-template-columns:1fr;gap:12px}.ti-panel{padding:15px}.ti-comparison{grid-template-columns:1fr}.ti-column+.ti-column{border-left:0;border-top:1px solid #ddd}.ti-workspace h1{font-size:26px}.ti-actions{align-items:stretch}.ti-actions input[type=text]{width:100%}.ti-actions button,.ti-upload{text-align:center;max-width:100%}}
    `}</style>
    <div className="ti-inner">
      <div className="ti-kicker">Investment thesis · External drafts</div>
      <h1>Bring your research into Charlie</h1>
      <p>Write with your subscription. Review here. Keep the same thesis history and future upgrade workflow.</p>
      <div className="ti-guide">
        <div><strong>01 · Prepare for ChatGPT</strong><span>Download a template with the current thesis and its starting version. Attach your own sources in ChatGPT.</span></div>
        <div><strong>02 · Import draft</strong><span>Upload the completed JSON file. Drafts stay in this inbox until you approve or dismiss them.</span></div>
        <div><strong>03 · Review changes</strong><span>Compare every section and source reference, then approve the new thesis or upgrade.</span></div>
      </div>
      <div className="ti-actions">
        <label>Ticker <input aria-label="Preparation ticker" type="text" value={ticker} maxLength={20} onChange={e=>setTicker(e.target.value.toUpperCase().trim())} disabled={busy}/></label>
        <button disabled={busy} onClick={()=>act(prepare)}>Prepare for ChatGPT</button>
        <label className="ti-upload">Import draft<input aria-label="Import draft file" type="file" accept=".json,application/json" disabled={busy} onChange={e=>{const file=e.target.files[0];e.target.value='';act(()=>upload(file));}}/></label>
        <button disabled={busy} onClick={()=>act(async()=>{await list(); if(draft)await select(draft.id);})}>Refresh inbox</button>
      </div>
      <p className="ti-muted">Preparation, import and approval make no paid model calls. Original documents stay where you keep them; references do not upload or verify the originals.</p>
      {busy && <p className="ti-busy" role="status">Working…</p>}
      {error && <div role="alert" className="ti-message ti-error">{error}</div>}
      {notice && <div role="status" className="ti-message">{notice}</div>}
      <section className="ti-panel"><h2>Draft inbox</h2>
        <p className="ti-muted">Most recent 100 matching drafts. Reimporting the same file opens its existing review.</p>
        <div className="ti-actions"><label>Filter by ticker <input type="text" aria-label="Inbox ticker filter" placeholder="All companies" maxLength={20} value={filter} disabled={busy} onChange={e=>setFilter(e.target.value.toUpperCase().trim())}/></label><button disabled={busy} onClick={()=>act(list)}>Find drafts</button></div>
        {loaded && !drafts.length && <p>No drafts yet for this view. Import a structured draft or prepare a company above.</p>}
        <div className="ti-inbox">{drafts.map(d=><button disabled={busy} key={d.id} aria-pressed={draft?.id===d.id} onClick={()=>act(()=>select(d.id))}><strong>{d.ticker} · {d.company}</strong><br/><span className="ti-status">{d.status}</span><br/><span className="ti-muted">{new Date(d.created_at).toLocaleString()}</span></button>)}</div>
      </section>
      {draft && <section className="ti-panel" aria-label="Draft review">
        <div className="ti-kicker">{draft.baseline ? 'Thesis upgrade' : 'Initial thesis'} · {draft.status}</div>
        <h2>{draft.ticker} — {draft.package.companyName}</h2>
        <p>{draft.sections.filter(s=>s.changed).length} of 4 sections changed. Read both versions, including removed items.</p>
        <div className="ti-message"><ul>{draft.warnings.map((w,i)=><li key={i}>{w}</li>)}</ul></div>
        {draft.stale && <div className="ti-message ti-error" role="alert">The live thesis has changed. Approval is blocked. Download a fresh preparation file, reconcile this draft in ChatGPT, and import the revised result.</div>}
        <div className="ti-actions"><button disabled={busy} onClick={()=>downloadFile(`${draft.ticker}-draft.json`, draft.package)}>Download this draft</button>
          {draft.status==='approved' && <button onClick={()=>onNavigate('portfolio',draft.ticker)}>Open saved thesis</button>}
        </div>
        {draft.sections.map(s=><section key={s.key} className="ti-section"><div className="ti-section-head"><h3>{s.name}</h3><span>{s.changed?'Changed':'Unchanged'}</span></div><div className="ti-comparison">
          <div className="ti-column"><div className="ti-column-title">Before · saved at import</div><Content value={s.before}/></div>
          <div className="ti-column"><div className="ti-column-title">Proposed</div><Content value={s.after}/></div>
        </div></section>)}
        <details><summary>Source register ({draft.package.sourceRegister.length}) · author-supplied metadata</summary><Content value={draft.package.sourceRegister}/></details>
        <details><summary>Authorship and preparation details · author supplied</summary><Content value={draft.package.provenance}/></details>
        {draft.status==='pending' && <section className="ti-panel ti-approval"><h2>Your approval</h2>
          <p>This will {draft.baseline?'replace the current detailed thesis':'create the initial thesis'} for {draft.ticker}. The reviewed draft is preserved, and upgrades retain the previous thesis for restoration.</p>
          <label className="ti-check"><input type="checkbox" checked={checked} disabled={busy||draft.stale} onChange={e=>setChecked(e.target.checked)}/><span>I reviewed the proposed thesis, signposts, risks and source limitations, and approve saving this version.</span></label>
          <div className="ti-actions"><label>Type {draft.ticker} to confirm <input aria-label="Confirm ticker" type="text" value={confirmation} disabled={busy||draft.stale} onChange={e=>setConfirmation(e.target.value.toUpperCase().trim())}/></label>
            <button className="ti-primary" disabled={!canApprove(draft,checked,confirmation,busy)} onClick={()=>act(approve)}>Approve & save thesis</button>
            <button disabled={busy} onClick={()=>act(async()=>{await call(`/${draft.id}/dismiss`,{});await select(draft.id);await list();})}>Dismiss draft</button>
          </div><p className="ti-muted">For edits, download this draft, revise it in ChatGPT, then import the new file. Paid research and report generation remain separate choices.</p>
        </section>}
      </section>}
    </div>
  </div>;
}
