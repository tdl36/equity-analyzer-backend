import * as React from 'react';
import {editionSelection,claimLabel,basisLabel,verifiedEditionPreview,editionReviewState,editionDecision,removedEditionClaims} from './onepager-edition-model.mjs';

export function OnepagerEditions({api,run,assignmentId,onOriginal}){
 const [pack,setPack]=React.useState(null),[rows,setRows]=React.useState([]),[selected,setSelected]=React.useState([]),[error,setError]=React.useState(''),[notice,setNotice]=React.useState(''),[busy,setBusy]=React.useState(false),[preview,setPreview]=React.useState(null),[reviewing,setReviewing]=React.useState(null),[ack,setAck]=React.useState(false);
 const alive=React.useRef(true),locked=React.useRef(false),pending=React.useRef(null),decisionAttempt=React.useRef(null);
 const url=api+`/api/research/assignments/${assignmentId}/onepager-editions`;
 async function read(path,body){const r=await fetch(path,{...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(30000)});let d;try{d=await r.json();}catch{throw Error('The response could not be read. Reload saved editions before retrying.');}if(!r.ok)throw Error(d.error||'Editions unavailable');return d;}
 async function load(){const [p,c]=await Promise.all([read(url),read(api+`/api/research/stock-analysis-run/${run.id}/corrections`)]);if(p.runId!==run.id||p.assignmentId!==assignmentId||p.ticker!==run.ticker||c.runId!==run.id)throw Error('The saved edition belongs to another report. Reopen its assignment.');if(alive.current){setPack(p);setRows(c.corrections);}}
 async function reload(){if(locked.current)return;setPreview(null);setReviewing(null);setAck(false);locked.current=true;setBusy(true);setError('');try{await load();}catch(e){if(alive.current)setError(e.message);}finally{locked.current=false;if(alive.current)setBusy(false);}}
 React.useEffect(()=>{alive.current=true;reload();return()=>{alive.current=false;};},[url,run.id]);
 async function save(){if(locked.current)return;setPreview(null);locked.current=true;setBusy(true);setError('');setNotice('');try{
  const value=editionSelection(pack,rows,selected,'');const signature=JSON.stringify(value);
  if(pending.current?.signature!==signature)pending.current={signature,body:{...value,requestId:crypto.randomUUID()}};
  const result=await read(url,pending.current.body);
  if(alive.current){pending.current=null;setSelected([]);setNotice(result.status==='pending'?`Pending edition ${result.id} saved. No correction has been accepted or applied.`:`Edition ${result.id} already has a saved ${result.status} decision. Reloaded its retained review.`);await load();}
 }catch(e){if(alive.current)setError(e.message+' Inspect saved editions before retrying. An unchanged selection keeps its submission identity.');}finally{locked.current=false;if(alive.current)setBusy(false);}}
 async function openPreview(id,layout){if(locked.current)return;locked.current=true;setBusy(true);setError('');setPreview(null);try{
  const value=verifiedEditionPreview(await read(url+`/${encodeURIComponent(id)}/preview?layout=${layout}`),id,assignmentId,run.id,run.ticker);
  if(alive.current)setPreview(value);
 }catch(e){if(alive.current)setError(e.message);}finally{locked.current=false;if(alive.current)setBusy(false);}}
 async function decide(action){if(locked.current||!reviewing||!ack)return;locked.current=true;setBusy(true);setError('');try{
  const body=editionDecision(reviewing,pack,action,'');const signature=JSON.stringify(body);
  if(decisionAttempt.current?.signature!==signature)decisionAttempt.current={signature,body:{...body,requestId:crypto.randomUUID()}};
  const result=await read(url+`/${reviewing.id}/decision`,decisionAttempt.current.body);
  if(result.decision?.editionId!==reviewing.id||result.decision?.action!==action)throw Error('Decision receipt does not match this edition. Reload saved history.');
  if(alive.current){setNotice(action==='accept'?'Editorial edition accepted. Its version-specific exports are now available. Original research and thesis are retained.':'Editorial edition rejected. The current accepted version is retained.');setReviewing(null);setPreview(null);decisionAttempt.current=null;await load();}
 }catch(e){if(alive.current)setError(e.message+' Reload history before making another decision; an unchanged retry keeps its identity.');}finally{locked.current=false;if(alive.current)setBusy(false);}}
 async function exportEdition(id,layout){if(locked.current)return;locked.current=true;setBusy(true);setError('');try{
  const response=await fetch(url+`/${encodeURIComponent(id)}/export?layout=${layout}`,{signal:AbortSignal.timeout(30000)});
  if(!response.ok){const d=await response.json();throw Error(d.error||'Edition export unavailable');}
  if(!response.headers.get('Content-Type')?.includes('text/html'))throw Error('Unexpected export format. Reload the edition.');
  const blob=await response.blob(),link=document.createElement('a'),href=URL.createObjectURL(blob);link.href=href;link.download=`${run.ticker}-edition-${id}-${layout}.html`;link.click();setTimeout(()=>URL.revokeObjectURL(href),1000);
  if(alive.current)setNotice('Accepted edition downloaded as a printable HTML document. Open it in your browser and print to PDF if needed.');
 }catch(e){if(alive.current)setError(e.message);}finally{locked.current=false;if(alive.current)setBusy(false);}}
 function evidence(p){return <details><summary>Inspect evidence and provenance</summary><p>{basisLabel(p.basis)} · {p.caveat}</p>{[p,...(p.additionalEvidence||[])].map((e,i)=>{const source=run.sources.find(s=>s.id===e.sourceId&&s.filename===e.filename&&s.originalHash===e.sha256);return <div key={i}><p>{e.filename} · {e.locator}</p><blockquote>{e.excerpt}</blockquote>{source?<button disabled={busy} onClick={()=>onOriginal(source)}>Download verified original</button>:<p>The original no longer matches this report. Inspect the saved source pack.</p>}<small>Original SHA-256 {e.sha256}</small></div>;})}<small>Claim {p.path}</small></details>;}

 let selectionError='';if(selected.length&&pack)try{editionSelection(pack,rows,selected,'');}catch(e){selectionError=e.message;}
 return <details className="sa-prepare"><summary>One-pager editions · review history</summary>
 <p>Group source-linked corrections into a saved edition for this assignment. Review the complete wording and evidence together, then open a labeled pending preview. Investor sign-in is required to accept or reject an edition. Accepted editions have separate printable exports.</p>
 <small>Assignment {assignmentId} · Report {run.id}</small>
 <button disabled={busy} onClick={reload}>Reload editions and corrections</button>
 {error&&<p role="alert">{error}</p>}{notice&&<p role="status">{notice}</p>}
 {!pack?<p>{busy?'Loading saved editions…':'Saved editions unavailable.'}</p>:<>
 <h4>Choose proposed changes</h4><p>Create corrections in “Propose a correction for review” above, then reload here. Showing up to 50 recent corrections; select at most 20, with one change per claim.</p>
 <fieldset disabled={busy}>{rows.map(c=><article key={c.id}>
 <label className="sa-check"><input type="checkbox" checked={selected.includes(c.id)} disabled={c.stale||c.fingerprint!==pack.fingerprint||(!selected.includes(c.id)&&selected.length>=20)} onChange={e=>setSelected(ids=>e.target.checked?[...ids,c.id]:ids.filter(id=>id!==c.id))}/><span>{claimLabel(c.proposal.path)}{c.stale||c.fingerprint!==pack.fingerprint?' · Baseline changed; prepare again':''}</span></label>
 <div className="sa-form-grid"><div><h5>Original</h5><p>{c.proposal.before===null?'New scenario input; no saved value.':c.proposal.before}</p></div><div><h5>Proposed · {basisLabel(c.proposal.basis)}</h5><p>{c.proposal.statement}</p></div></div><p>{c.proposal.reason}</p>{evidence(c.proposal)}</article>)}
 {!rows.length&&<p>No pending corrections have been saved for this report.</p>}
 {selectionError&&<p role="alert">{selectionError}</p>}<button disabled={!selected.length||!!selectionError} onClick={save}>{busy?'Working…':`Save pending edition (${selected.length}/20)`}</button></fieldset>
 <h4>Saved editions</h4><p>Showing up to {pack.limit} most recent editions. Saving an edition alone does not approve it; any recorded decision is shown separately. Original availability and usage restrictions are rechecked when saving; opening this list does not reverify them.</p>
 {pack.editions.map(e=>{const review=editionReviewState(e);return <article key={e.id}><h5>{review.label} · {new Date(e.created_at).toLocaleString()}</h5><small>Edition {e.id}{e.stale?' · Baseline or parent changed; new decisions require fresh preparation':''}</small>{e.decision?<p>Investor decision recorded {new Date(e.decision.decidedAt).toLocaleString()}. Original research remains unchanged. Acceptance records editorial review; it does not certify accuracy or approve a thesis.</p>:<div><button disabled={busy||!review.canPreview} onClick={()=>openPreview(e.id,'illustrated')}>Preview pending investment map</button><button disabled={busy||!review.canPreview} onClick={()=>openPreview(e.id,'brief')}>Preview pending brief</button></div>}{e.decision?.action==='accept'&&<div><button disabled={busy||e.stale} onClick={()=>exportEdition(e.id,'illustrated')}>Download accepted investment map</button><button disabled={busy||e.stale} onClick={()=>exportEdition(e.id,'brief')}>Download accepted brief</button></div>}{review.canPreview&&pack.canDecide&&<button disabled={busy} onClick={()=>{setReviewing(e);setAck(false);}}>Review edition decision</button>}{e.record.claims.map(c=><details key={c.correctionId}><summary>{claimLabel(c.proposal.path)}</summary><div className="sa-form-grid"><div><h5>Original</h5><p>{c.proposal.before===null?'New scenario input; no saved value.':c.proposal.before}</p></div><div><h5>Proposed · {basisLabel(c.proposal.basis)}</h5><p>{c.proposal.statement}</p></div></div><p>{c.proposal.reason}</p>{evidence(c.proposal)}<small>Correction {c.correctionId}</small></details>)}</article>})}
 {reviewing&&<section aria-label="Review editorial decision"><h4>Review this exact edition</h4><p>Edition {reviewing.id}. This complete selection replaces the previous editorial selection; it does not add to it. Original research and thesis stay unchanged.</p>
 {reviewing.record.claims.map(c=><article key={c.correctionId}><h5>{claimLabel(c.proposal.path)}</h5><p>Original: {c.proposal.before===null?'New scenario input; no saved value.':c.proposal.before}</p><p>Proposed: {c.proposal.statement}</p>{evidence(c.proposal)}</article>)}
 <h5>Changes removed from the previous accepted edition</h5>{removedEditionClaims(reviewing,pack).length?removedEditionClaims(reviewing,pack).map(c=><p key={c.correctionId}>{claimLabel(c.proposal.path)}: {c.proposal.statement}. This correction will no longer be included; the selected replacement or original wording will be used.</p>):<p>None.</p>}
 <label className="sa-check"><input type="checkbox" disabled={busy} checked={ack} onChange={e=>setAck(e.target.checked)}/>I reviewed the wording, evidence, qualifications and any removed corrections for this exact edition.</label>
 <button disabled={busy||!ack} onClick={()=>decide('accept')}>Accept this editorial edition</button><button disabled={busy||!ack} onClick={()=>decide('reject')}>Reject this edition</button><button disabled={busy} onClick={()=>setReviewing(null)}>Close review</button></section>}
 {preview&&<section aria-label="Pending edition preview"><h4>Pending preview · not accepted</h4><small>Edition {preview.id} · {preview.layout==='brief'?'Brief':'Investment map'}</small><p>{preview.caveat}</p><button onClick={()=>setPreview(null)}>Close pending preview</button><iframe title={`${run.ticker} pending edition ${preview.id}`} sandbox="" srcDoc={preview.html} style={{width:'100%',height:850,border:'1px solid #ccd7de',background:'white'}}/></section>}
 {!pack.editions.length&&<p>No saved editions for this assignment.</p>}</>}
 </details>;
}
