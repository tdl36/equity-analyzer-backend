import * as React from 'react';
import {editionSelection,claimLabel,basisLabel} from './onepager-edition-model.mjs';

export function OnepagerEditions({api,run,assignmentId,onOriginal}){
 const [pack,setPack]=React.useState(null),[rows,setRows]=React.useState([]),[selected,setSelected]=React.useState([]),[error,setError]=React.useState(''),[notice,setNotice]=React.useState(''),[busy,setBusy]=React.useState(false);
 const alive=React.useRef(true),locked=React.useRef(false),pending=React.useRef(null);
 const url=api+`/api/research/assignments/${assignmentId}/onepager-editions`;
 async function read(path,body){const r=await fetch(path,{...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(30000)});let d;try{d=await r.json();}catch{throw Error('The response could not be read. Reload saved editions before retrying.');}if(!r.ok)throw Error(d.error||'Editions unavailable');return d;}
 async function load(){const [p,c]=await Promise.all([read(url),read(api+`/api/research/stock-analysis-run/${run.id}/corrections`)]);if(p.runId!==run.id||p.assignmentId!==assignmentId||p.ticker!==run.ticker||c.runId!==run.id)throw Error('The saved edition belongs to another report. Reopen its assignment.');if(alive.current){setPack(p);setRows(c.corrections);}}
 async function reload(){if(locked.current)return;locked.current=true;setBusy(true);setError('');try{await load();}catch(e){if(alive.current)setError(e.message);}finally{locked.current=false;if(alive.current)setBusy(false);}}
 React.useEffect(()=>{alive.current=true;reload();return()=>{alive.current=false;};},[url,run.id]);
 async function save(){if(locked.current)return;locked.current=true;setBusy(true);setError('');setNotice('');try{
  const value=editionSelection(pack,rows,selected,'');const signature=JSON.stringify(value);
  if(pending.current?.signature!==signature)pending.current={signature,body:{...value,requestId:crypto.randomUUID()}};
  const result=await read(url,pending.current.body);
  if(alive.current){pending.current=null;setSelected([]);setNotice(`Pending edition ${result.id} saved. No correction has been accepted or applied.`);await load();}
 }catch(e){if(alive.current)setError(e.message+' Inspect saved editions before retrying. An unchanged selection keeps its submission identity.');}finally{locked.current=false;if(alive.current)setBusy(false);}}
 function evidence(p){const source=run.sources.find(s=>s.id===p.sourceId&&s.filename===p.filename&&s.originalHash===p.sha256);return <details><summary>Inspect evidence and provenance</summary><p>{basisLabel(p.basis)} · {p.filename} · {p.locator}</p><blockquote>{p.excerpt}</blockquote><p>{p.caveat}</p>{source?<button disabled={busy} onClick={()=>onOriginal(source)}>Download verified original</button>:<p>The original no longer matches this report. Inspect the saved source pack.</p>}<small>Claim {p.path} · Original SHA-256 {p.sha256}</small></details>;}
 let selectionError='';if(selected.length&&pack)try{editionSelection(pack,rows,selected,'');}catch(e){selectionError=e.message;}
 return <details className="sa-prepare"><summary>One-pager editions · pending review</summary>
 <p>Group source-linked corrections into a saved edition for this assignment. Review the complete wording and evidence together. Acceptance and revised exports are not available yet.</p>
 <small>Assignment {assignmentId} · Report {run.id}</small>
 <button disabled={busy} onClick={reload}>Reload editions and corrections</button>
 {error&&<p role="alert">{error}</p>}{notice&&<p role="status">{notice}</p>}
 {!pack?<p>{busy?'Loading saved editions…':'Saved editions unavailable.'}</p>:<>
 <h4>Choose proposed changes</h4><p>Create corrections in “Propose a correction for review” above, then reload here. Showing up to 50 recent corrections; select at most 20, with one change per claim.</p>
 <fieldset disabled={busy}>{rows.map(c=><article key={c.id}>
 <label className="sa-check"><input type="checkbox" checked={selected.includes(c.id)} disabled={c.stale||c.fingerprint!==pack.fingerprint||(!selected.includes(c.id)&&selected.length>=20)} onChange={e=>setSelected(ids=>e.target.checked?[...ids,c.id]:ids.filter(id=>id!==c.id))}/><span>{claimLabel(c.proposal.path)}{c.stale||c.fingerprint!==pack.fingerprint?' · Baseline changed; prepare again':''}</span></label>
 <div className="sa-form-grid"><div><h5>Original</h5><p>{c.proposal.before}</p></div><div><h5>Proposed · {basisLabel(c.proposal.basis)}</h5><p>{c.proposal.statement}</p></div></div><p>{c.proposal.reason}</p>{evidence(c.proposal)}</article>)}
 {!rows.length&&<p>No pending corrections have been saved for this report.</p>}
 {selectionError&&<p role="alert">{selectionError}</p>}<button disabled={!selected.length||!!selectionError} onClick={save}>{busy?'Saving…':`Save pending edition (${selected.length}/20)`}</button></fieldset>
 <h4>Saved editions</h4><p>Showing up to {pack.limit} most recent editions. A saved edition is a review record, not an approved one-pager. Original availability and usage restrictions are rechecked when saving; opening this list does not reverify them.</p>
 {pack.editions.map(e=><article key={e.id}><h5>Pending edition · {new Date(e.created_at).toLocaleString()}</h5><small>Edition {e.id}{e.stale?' · Baseline changed; prepare a new edition':''}</small>{e.record.claims.map(c=><details key={c.correctionId}><summary>{claimLabel(c.proposal.path)}</summary><div className="sa-form-grid"><div><h5>Original</h5><p>{c.proposal.before}</p></div><div><h5>Proposed · {basisLabel(c.proposal.basis)}</h5><p>{c.proposal.statement}</p></div></div><p>{c.proposal.reason}</p>{evidence(c.proposal)}<small>Correction {c.correctionId}</small></details>)}</article>)}
 {!pack.editions.length&&<p>No saved editions for this assignment.</p>}</>}
 </details>;
}
