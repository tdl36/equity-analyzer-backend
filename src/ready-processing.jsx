import * as React from 'react';
import {parseTimestamp} from './workspace-model.mjs';
import {CollectionSynthesisStatus} from './collection-synthesis';
export function ReadyProcessing({api}) {
 const [rows,setRows]=React.useState([]),[selected,setSelected]=React.useState(new Set()),[query,setQuery]=React.useState('');
 const [error,setError]=React.useState(''),[message,setMessage]=React.useState(''),[loaded,setLoaded]=React.useState(false),[busy,setBusy]=React.useState(false);
 const alive=React.useRef(false),lock=React.useRef(false),serial=React.useRef(0);
 const refresh=async()=>{const generation=++serial.current;try{
  const r=await fetch(`${api}/api/catalysts/proposals`,{signal:AbortSignal.timeout(20000)});if(!r.ok)throw Error('Could not load ready jobs.');
  const d=await r.json();if(!Array.isArray(d.proposals))throw Error('Ready-job response is incomplete.');
  if(alive.current&&generation===serial.current){setRows(d.proposals);setSelected(old=>new Set([...old].filter(id=>d.proposals.some(p=>p.jobId===id))));setError('');setLoaded(true);}
 }catch(e){if(alive.current&&generation===serial.current)setError(`${e.message} Selection is disabled until a fresh list is available.`);}};
 React.useEffect(()=>{alive.current=true;refresh();const timer=setInterval(()=>{if(!document.hidden&&!lock.current)refresh();},30000);return()=>{alive.current=false;serial.current++;clearInterval(timer);};},[api]);
 const visible=rows.filter(p=>`${p.ticker} ${p.topic}`.toLowerCase().includes(query.trim().toLowerCase()));
 const start=async()=>{
  if(lock.current||error||!selected.size)return;lock.current=true;setBusy(true);setMessage('');serial.current++;
  const ids=rows.filter(p=>selected.has(p.jobId)).map(p=>p.jobId);
  try{
   const r=await fetch(`${api}/api/catalysts/proposals/approve`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({jobIds:ids}),signal:AbortSignal.timeout(30000)});
   const d=await r.json();if(!r.ok||!Number.isInteger(d.approved))throw Error('Submission was not confirmed.');
   if(alive.current)setMessage(`${d.approved} job${d.approved===1?'':'s'} queued for synthesis. ${ids.length-d.approved>0?`${ids.length-d.approved} selections were already handled or no longer ready. `:''}Results remain drafts for review.`);
  }catch(e){if(alive.current)setMessage('Submission could not be confirmed. Check Catalyst notes for queued or running jobs before submitting again. Charlie only queues jobs still awaiting processing approval.');}
  finally{if(alive.current){setSelected(new Set());await refresh();setBusy(false);}lock.current=false;}
 };
 return <section className="workspace-panel" aria-label="Ready to process"><p className="workspace-eyebrow">YOU CHOOSE WHAT RUNS</p><h2>Ready to process</h2>
  <p className="workspace-lead">Select the source folders you want Charlie to synthesize.</p>
  <p>These are detected event folders awaiting processing approval. Starting selected jobs uses model credits and the Mac agent. It does not enable ongoing automatic synthesis or approve the resulting research.</p>
  <div className="desk-filter"><label>Find ready jobs<input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Ticker or event"/></label><button onClick={refresh} disabled={busy}>Refresh ready jobs</button></div>
  {error&&<p role="alert" className="workspace-error">{error}</p>}{message&&<p role="status" className="workspace-notice">{message} <a href="#view=catalysts">Track synthesis & results →</a></p>}
  {!loaded&&!error&&<p>Loading ready jobs…</p>}
  <fieldset disabled={busy||!!error||!loaded} style={{border:0,padding:0,minWidth:0}}><legend>{visible.length} shown · {rows.length} ready · {selected.size} selected</legend>
   <div className="desk-filter"><button onClick={()=>setSelected(new Set([...selected,...visible.map(p=>p.jobId)]))} disabled={!visible.length}>Select shown</button><button onClick={()=>setSelected(new Set())} disabled={!selected.size}>Clear selection</button></div>
   {visible.map(p=><label className="workspace-panel" key={p.jobId} style={{display:'flex',gap:14,alignItems:'flex-start',cursor:'pointer',marginBottom:10}}>
    <input type="checkbox" aria-label={`Process ${p.ticker} ${p.topic}`} checked={selected.has(p.jobId)} onChange={e=>setSelected(old=>{const next=new Set(old);if(e.target.checked)next.add(p.jobId);else next.delete(p.jobId);return next;})}/>
    <span style={{minWidth:0,overflowWrap:'anywhere'}}><strong>{p.ticker} · {p.topic||'Event folder'}</strong><span style={{display:'block'}}>{Number.isInteger(p.fileCount)?`${p.fileCount} detected source files`:'Source count unavailable'} · {p.length||'standard'} synthesis</span><small>Detected {parseTimestamp(p.proposedAt)?.toLocaleString()||'date unavailable'} · CATALYSTS folder</small></span>
   </label>)}
   {loaded&&!rows.length&&!error&&<p>No detected event folders are awaiting processing approval. Collection can continue while automatic synthesis is off.</p>}
   {rows.length>0&&!visible.length&&<p>No ready jobs match this search.</p>}
   <button className="workspace-primary" onClick={start} disabled={!selected.size}>{busy?'Submitting…':`Approve & process ${selected.size} selected job${selected.size===1?'':'s'}`}</button>
  </fieldset>
  <p className="desk-explainer">Selection applies to every checked job, including those hidden by your search. The list shows the latest 100 proposals; source counts reflect detection, and the agent validates readable files when processing.</p>
  <div className="desk-filter"><a href="#view=catalysts">Queued jobs & synthesis results →</a><a href="#view=analysts">Analyst tasks & draft review →</a><a href="#view=pipelinelab">Choose STOCKS sources & outputs in Lab →</a></div>
  <p className="desk-explainer">An imported document alone is not a ready synthesis job. Lab source libraries, analyst tasks and completed drafts keep their own output and review controls.</p>
  <CollectionSynthesisStatus api={api}/>
 </section>;
}
