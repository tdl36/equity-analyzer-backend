import React from 'react';

const labels={brief:'Executive brief',takeaways:'Key takeaways',record:'Management record',questions:'Follow-up questions',assessment:'Investment assessment'};
const originalKeys={brief:'brief',takeaways:'summary',record:'meeting_summary',questions:'questions',assessment:'assessment'};
export function SummaryComparison({summary,api,getKey,renderHtml}) {
 const [open,setOpen]=React.useState(false),[rows,setRows]=React.useState([]),[selected,setSelected]=React.useState(''),[section,setSection]=React.useState('takeaways'),[busy,setBusy]=React.useState(false),[error,setError]=React.useState(''),[feedback,setFeedback]=React.useState(''),[saved,setSaved]=React.useState(false);
 const epoch=React.useRef(0);
 React.useEffect(()=>{epoch.current++;setRows([]);setSelected('');setOpen(false);setError('');},[summary.id]);
 const base=`${api}/api/summaries/${encodeURIComponent(summary.id)}/comparisons`;
 React.useEffect(()=>{if(!open)return;let alive=true;let timer;
 async function poll(){try{const r=await fetch(base,{signal:AbortSignal.timeout(20000)});const d=await r.json();if(!r.ok)throw Error(d.error||'Comparison could not be loaded.');if(alive){setRows(d.comparisons);setError('');}}catch(e){if(alive)setError(e.message);}finally{if(alive)timer=setTimeout(poll,8000);}}
 poll();return()=>{alive=false;clearTimeout(timer);};},[base,open]);
 const row=rows.find(r=>r.id===selected)||rows[0];
 React.useEffect(()=>{setFeedback(row?.feedback||'');setSaved(false);},[row?.id]);
 const state=row?.state||{};
 async function start(resumeId){const token=epoch.current;setBusy(true);setError('');try{const r=await fetch(base,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({apiKey:getKey(),resumeId}),signal:AbortSignal.timeout(25000)});const d=await r.json();if(!r.ok)throw Error(d.error||'Could not start comparison.');if(token!==epoch.current)return;setSelected(d.id);const fresh=await fetch(base);if(fresh.ok&&token===epoch.current)setRows((await fresh.json()).comparisons);}catch(e){if(token===epoch.current)setError(e.message);}finally{if(token===epoch.current)setBusy(false);}}
 const record=Object.keys(state.parts||{}).sort((a,b)=>Number(a)-Number(b)).map(k=>state.parts[k].record).join('\n\n');
 const text=section==='record'?record:state.sections?.[section];
 return <section className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 my-4">
 <button className="text-left w-full flex justify-between gap-3" aria-expanded={open} onClick={()=>setOpen(!open)}><span><strong>Compare improved notes</strong><small className="block text-slate-400 mt-1">Trial workspace · Your original note stays intact</small></span><span>{open?'−':'+'}</span></button>
 {open&&<div className="mt-4 space-y-4">
 <p className="text-sm text-slate-400">Generate an alternative from the complete saved source. Long transcripts are read in parts, with every part retained in the management record. This starts paid AI generation; it does not transcribe the audio again.</p>
 <button className="px-3 py-2 rounded-lg bg-amber-600 text-white disabled:opacity-50" disabled={busy||!summary.rawNotes?.trim()} onClick={()=>start()}>{busy?'Starting…':rows.length?'Open or generate comparison for current source':'Generate improved comparison'}</button>
 {!summary.rawNotes?.trim()&&<p role="status">No saved source text is available for this note yet.</p>}
 {error&&<p role="alert" className="text-red-400">{error}</p>}
 {row&&<>
 {rows.length>1&&<label className="block">Comparison version <select className="bg-slate-900 p-2 rounded" value={row.id} onChange={e=>setSelected(e.target.value)}>{rows.map(r=><option key={r.id} value={r.id}>{r.version} · {new Date(r.created_at).toLocaleString()}</option>)}</select></label>}
 <div role="status" className="text-sm"><strong>{row.status==='complete'?'Ready for comparison':row.status==='failed'?'Needs retry':state.progress||'Queued'}</strong><p className="text-slate-400">{(state.coveredCharacters||0).toLocaleString()} / {(state.sourceCharacters||summary.rawNotes?.length||0).toLocaleString()} source characters processed · {Object.keys(state.parts||{}).length} / {state.totalParts||'—'} parts saved · Last update {new Date(row.updated_at).toLocaleTimeString()}</p></div>
 {row.error&&<p className="text-red-400">{row.error}</p>}
 {row.status!=='complete'&&<div><button disabled={busy} className="underline" onClick={()=>start(row.id)}>Resume saved comparison</button><p className="text-xs text-slate-400">Safe to resume after interruption; an active worker cannot be started twice. Saved sections are reused.</p></div>}
 <p className="text-xs text-slate-400">Left: original output frozen when this comparison began. Right: {row.version}. No prior thesis/model comparison or independent factual verification. {state.hierarchicalSynthesis?'Long-source synthesis uses consolidated evidence; inspect the full part records for detail.':''}</p>
 <div className="flex flex-wrap gap-2" aria-label="Comparison section">{Object.entries(labels).map(([key,label])=><button key={key} aria-pressed={section===key} className={`px-3 py-2 rounded-lg text-sm ${section===key?'bg-amber-600 text-white':'bg-white/5'}`} onClick={()=>setSection(key)}>{label}</button>)}</div>
 <div className="grid grid-cols-1 xl:grid-cols-2 gap-4"><article className="min-w-0 rounded-lg border border-white/10 p-4"><h4 className="font-semibold mb-3">Original · {labels[section]}</h4><div className="prose prose-invert max-w-none text-sm break-words" dangerouslySetInnerHTML={{__html:renderHtml(row.baseline?.[originalKeys[section]]||'<p>No original section saved.</p>')}}/></article><article className="min-w-0 rounded-lg border border-amber-500/20 p-4"><h4 className="font-semibold mb-3">Improved · {labels[section]}</h4><div className="whitespace-pre-wrap break-words text-sm leading-relaxed">{text||'This section has not been generated yet. Saved management-record parts are available as processing progresses.'}</div></article></div>
 <label className="block text-sm font-medium">Comparison notes<textarea className="block w-full bg-transparent border border-white/20 rounded-lg p-3 mt-2" rows={3} maxLength={10000} value={feedback} onChange={e=>{setFeedback(e.target.value);setSaved(false);}} placeholder="What is better? What detail was lost? Which version would you use?"/></label>
 <button className="underline text-sm" onClick={async()=>{try{const r=await fetch(`${base}/${row.id}/feedback`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({feedback})});if(!r.ok)throw Error('Comparison notes could not be saved.');setSaved(true);}catch(e){setError(e.message);}}}>Save comparison notes</button>{saved&&<span role="status" className="text-sm ml-3">Saved</span>}
 </>}
 </div>}
 </section>;
}
