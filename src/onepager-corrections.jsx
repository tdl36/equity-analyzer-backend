import * as React from 'react';

export function OnepagerCorrections({api,run,onOriginal}){
 const [pack,setPack]=React.useState(null),[error,setError]=React.useState(''),[busy,setBusy]=React.useState(false);
 const [path,setPath]=React.useState(''),[statement,setStatement]=React.useState(''),[basis,setBasis]=React.useState('interpretation'),[reason,setReason]=React.useState(''),[sourceId,setSource]=React.useState(''),[excerpt,setExcerpt]=React.useState(''),[locator,setLocator]=React.useState('');
 const [additionalEvidence,setAdditionalEvidence]=React.useState([]);
 const locked=React.useRef(false),pending=React.useRef(null),alive=React.useRef(true);
 const url=api+`/api/research/stock-analysis-run/${run.id}/corrections`;
 async function call(body){const r=await fetch(url,{...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(30000)});const d=await r.json();if(!r.ok)throw Error(d.error||'Corrections unavailable');return d;}
 async function load(){try{const d=await call();if(alive.current){setPack(d);setError('');}}catch(e){if(alive.current)setError(e.message);}}
 React.useEffect(()=>{alive.current=true;load();return()=>{alive.current=false;};},[url]);
 async function submit(){if(locked.current)return;locked.current=true;setBusy(true);setError('');try{
  const value={fingerprint:pack.fingerprint,path,statement,basis,reason,sourceId,excerpt,locator,...(additionalEvidence.length?{additionalEvidence}:{})};const signature=JSON.stringify(value);
  if(pending.current?.signature!==signature)pending.current={signature,body:{...value,requestId:crypto.randomUUID()}};
  await call(pending.current.body);if(alive.current){pending.current=null;setStatement('');setReason('');setExcerpt('');setLocator('');setAdditionalEvidence([]);await load();}
 }catch(e){if(alive.current)setError(e.message+' Inspect saved corrections before retrying; unchanged form data retains its submission identity.');}finally{locked.current=false;if(alive.current)setBusy(false);}}
 const selected=pack?.fields.find(f=>f.path===path);
 const title=p=>p.replace(/^\//,'').split('/').map(x=>/^\d+$/.test(x)?String(Number(x)+1):x.replaceAll('_',' ')).join(' · ');
 return <details className="sa-prepare"><summary>Propose a correction for review</summary>
 <p>Save proposed wording separately from this report. The passage is checked against the frozen source and original file. Meaning and completeness still need investor review. These pending records cannot be applied or published here; the current report, one-pager and thesis stay unchanged.</p>
 {error&&<p role="alert">{error}</p>}<button disabled={busy} onClick={load}>Reload saved corrections</button>
 {!pack?<p>Loading saved evidence…</p>:<><fieldset disabled={busy}>
 <label>Claim to correct<select value={path} onChange={e=>setPath(e.target.value)}><option value="">Choose a saved claim</option>{pack.fields.map(f=><option key={f.path} value={f.path}>{title(f.path)}</option>)}</select></label>
 {selected&&<><h5>Original wording</h5><p style={{whiteSpace:'pre-wrap'}}>{selected.before||'Unavailable'}</p></>}
 <label>Proposed wording<textarea value={statement} onChange={e=>setStatement(e.target.value)} maxLength={2000} rows={4}/></label>
 <label>Claim type<select value={basis} onChange={e=>setBasis(e.target.value)}>{[['reported_fact','Reported fact'],['management_guidance','Management guidance'],['broker_estimate','Broker estimate'],['interpretation','Interpretation'],['hypothesis','Hypothesis']].map(([v,t])=><option key={v} value={v}>{t}</option>)}</select></label>
 <label>Why this correction is needed<textarea value={reason} onChange={e=>setReason(e.target.value)} maxLength={2000} rows={2}/></label>
 <label>Frozen source<select value={sourceId} onChange={e=>setSource(e.target.value)}><option value="">Choose an original</option>{pack.sources.map(s=><option key={s.id} value={s.id}>{s.filename}</option>)}</select></label>
 {sourceId&&<button type="button" onClick={()=>onOriginal(run.sources.find(s=>s.id===sourceId))}>Download verified original</button>}
 <label>Exact supporting passage<textarea value={excerpt} onChange={e=>setExcerpt(e.target.value)} maxLength={4000} rows={4}/></label>
 <label>Page or section locator<input value={locator} onChange={e=>setLocator(e.target.value)} maxLength={200}/></label>
 {additionalEvidence.map((e,i)=><div key={i} style={{borderTop:'1px solid #ddd',paddingTop:12,marginTop:12}}>
 <h5>Additional original {i+1}</h5>
 <label>Frozen source<select value={e.sourceId} onChange={event=>setAdditionalEvidence(items=>items.map((x,j)=>j===i?{...x,sourceId:event.target.value}:x))}><option value="">Choose another original</option>{pack.sources.map(x=><option key={x.id} value={x.id}>{x.filename}</option>)}</select></label>
 {e.sourceId&&<button type="button" onClick={()=>onOriginal(run.sources.find(x=>x.id===e.sourceId))}>Download verified original</button>}
 <label>Exact supporting passage<textarea rows={4} maxLength={4000} value={e.excerpt} onChange={event=>setAdditionalEvidence(items=>items.map((x,j)=>j===i?{...x,excerpt:event.target.value}:x))}/></label>
 <label>Page or section locator<input maxLength={200} value={e.locator} onChange={event=>setAdditionalEvidence(items=>items.map((x,j)=>j===i?{...x,locator:event.target.value}:x))}/></label>
 <button type="button" onClick={()=>setAdditionalEvidence(items=>items.filter((_,j)=>j!==i))}>Remove additional original</button>
 </div>)}
 <button type="button" disabled={additionalEvidence.length>=7} onClick={()=>setAdditionalEvidence(items=>[...items,{sourceId:'',excerpt:'',locator:''}])}>Add another supporting original</button>
 <button disabled={!path||!statement.trim()||!reason.trim()||!sourceId||!excerpt.trim()||!locator.trim()||additionalEvidence.some(e=>!e.sourceId||!e.excerpt.trim()||!e.locator.trim())} onClick={submit}>{busy?'Saving…':'Save pending correction'}</button>
 </fieldset><h4>Pending corrections</h4><p>Showing up to {pack.limit} most recent records. No correction is accepted automatically.</p>
 {pack.corrections.length?pack.corrections.map(c=><article key={c.id} style={{overflowWrap:'anywhere'}}><h5>{title(c.proposal.path)}</h5><small>{new Date(c.created_at).toLocaleString()} · Pending investor review{c.stale?' · Baseline changed; prepare again':''}</small><div className="sa-form-grid"><div><h5>Original</h5><p style={{whiteSpace:'pre-wrap'}}>{c.proposal.before}</p></div><div><h5>Proposed</h5><p style={{whiteSpace:'pre-wrap'}}>{c.proposal.statement}</p></div></div><p>{c.proposal.reason}</p><details><summary>Inspect supporting evidence</summary>{[c.proposal,...(c.proposal.additionalEvidence||[])].map((e,i)=><div key={i}><p>{e.filename} · {e.locator}</p><blockquote>{e.excerpt}</blockquote></div>)}<p>{c.proposal.caveat}</p><small>Correction {c.id} · {c.proposal.path}</small></details></article>):<p>No saved corrections for this report.</p>}</>}
 </details>;
}
