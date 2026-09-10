import React from 'react';

const sections=[['brief','Executive brief','brief'],['takeaways','Key takeaways','summary'],['record','Management record','meeting_summary'],['questions','Follow-up questions','questions'],['assessment','Investment assessment','assessment']];

export function SummaryComparison({summary,api,getKey,renderHtml,children}) {
 const [view,setView]=React.useState('original');
 const [rows,setRows]=React.useState([]),[loaded,setLoaded]=React.useState(false),[selected,setSelected]=React.useState('');
 const [busy,setBusy]=React.useState(false),[error,setError]=React.useState(''),[feedback,setFeedback]=React.useState(''),[saved,setSaved]=React.useState(false);
 const [exportBusy,setExportBusy]=React.useState(''),[exportMessage,setExportMessage]=React.useState('');
 const epoch=React.useRef(0);
 const base=`${api}/api/summaries/${encodeURIComponent(summary.id)}/comparisons`;
 React.useEffect(()=>{epoch.current++;setRows([]);setLoaded(false);setSelected('');setView('original');setError('');},[summary.id]);
 React.useEffect(()=>{let alive=true,timer;
  async function poll(){try{const r=await fetch(base,{signal:AbortSignal.timeout(20000)});const d=await r.json();if(!r.ok)throw Error(d.error||'Improved notes could not be loaded.');if(alive){setRows(d.comparisons);setLoaded(true);setError('');}}catch(e){if(alive)setError(e.message);}finally{if(alive)timer=setTimeout(poll,8000);}}
  poll();return()=>{alive=false;clearTimeout(timer);};
 },[base]);
 const row=rows.find(r=>r.id===selected)||rows[0],state=row?.state||{};
 React.useEffect(()=>{setFeedback(row?.feedback||'');setSaved(false);},[row?.id]);
 const running=row&&['queued','running'].includes(row.status);
 const stale=running&&Date.now()-new Date(row.updated_at).getTime()>10*60*1000;
 async function start(){const token=epoch.current;setBusy(true);setError('');try{
  const r=await fetch(base,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({apiKey:getKey(),resumeId:row?.id}),signal:AbortSignal.timeout(25000)});
  const d=await r.json();if(!r.ok)throw Error(d.error||'Could not start improved notes.');if(token!==epoch.current)return;setSelected(d.id);
  const fresh=await fetch(base,{signal:AbortSignal.timeout(20000)});if(!fresh.ok)throw Error('Could not refresh note status.');const data=await fresh.json();if(token===epoch.current)setRows(data.comparisons);
 }catch(e){if(token===epoch.current)setError(e.message);}finally{if(token===epoch.current)setBusy(false);}}
 const record=Object.keys(state.parts||{}).sort((a,b)=>Number(a)-Number(b)).map(k=>state.parts[k].record).join('\n\n');
 const escapeHtml=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 async function exportNotes(action,key='all') {
  if(!row||row.status!=='complete'||exportBusy)return;
  setExportBusy(`${action}:${key}`);setExportMessage('');
  const chosen=sections.filter(([k])=>key==='all'||k===key);
  const title=`Improved — ${row.baseline?.title||summary.title||'Meeting notes'}`;
  const label=key==='all'?'All sections':chosen[0][1];
  const texts=chosen.map(([k,label])=>[label,k==='record'?record:state.sections?.[k]||'']);
  try {
   if(texts.some(([,text])=>!text.trim()))throw Error('The requested section is not available.');
   if(action==='copy') {
    await navigator.clipboard.writeText(`${title}\n${label}\n\n`+texts.map(([label,text])=>`${label}\n\n${text}`).join('\n\n---\n\n'));
    setExportMessage('Improved notes copied.');return;
   }
   let url,body;
   if(action==='email') {
    let creds;try{creds=JSON.parse(localStorage.getItem('emailCredentials')||'null');}catch{}
    if(!creds?.email)throw Error('Set your email credentials in Settings first.');
    url=`${api}/api/email-summary-section`;
    body={email:creds.email,subject:`${title} — ${label}`,section:'improved',title:escapeHtml(title),topic:escapeHtml(summary.topic||'General'),
     content:texts.map(([label,text])=>`<h2>${escapeHtml(label)}</h2>`+text.split('\n\n').map(p=>`<p>${escapeHtml(p).replace(/\n/g,'<br>')}</p>`).join('')).join(''),
     smtpConfig:{use_gmail:creds.useGmail,gmail_user:creds.gmailUser,gmail_app_password:creds.gmailPassword,from_email:creds.gmailUser}};
   } else {
    url=action==='icloud'?`${api}/api/summaries/${encodeURIComponent(summary.id)}/save-to-icloud`:`${api}/api/summary-section-to-docx`;
    body={summaryId:summary.id,comparisonId:row.id,section:key==='record'?'meeting':key};
   }
   const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(60000)});
   const d=await r.json();if(!r.ok)throw Error(d.error||'Export failed.');
   if(action==='word') {
    const bytes=Uint8Array.from(atob(d.fileData),c=>c.charCodeAt(0));
    const downloadUrl=URL.createObjectURL(new Blob([bytes],{type:'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}));
    const link=document.createElement('a');link.href=downloadUrl;link.download=d.filename;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(downloadUrl),60000);
    setExportMessage('Improved Word document downloaded.');
   } else setExportMessage(action==='icloud'?`Queued for iCloud: ${d.filename}. Your connected Mac agent will save it in SUMMARIES/Word Exports.`:'Improved notes emailed.');
  } catch(e) {setExportMessage((e.name==='TimeoutError'||e.name==='AbortError')?'No confirmation received. Check your email or iCloud export queue before retrying to avoid duplicates.':e.message);}
  finally{setExportBusy('');}
 }
 function controls(key='all') {return <div className="flex flex-wrap gap-2" aria-label={`Improved ${key} export controls`}>{[['icloud','Save to iCloud'],['word','Download Word'],['copy','Copy'],['email','Email me']].map(([action,label])=><button key={action} type="button" disabled={!!exportBusy||row?.status!=='complete'} onClick={()=>exportNotes(action,key)} className="px-3 py-2 rounded-lg text-xs font-medium bg-white/10 border border-white/15 hover:bg-white/15 disabled:opacity-50" aria-label={`${label} — Improved ${key}`}>{exportBusy===`${action}:${key}`?'Working…':label}</button>)}</div>;}
 const status=row?.status==='complete'?'Ready':row?.status==='failed'?'Needs retry':running?'Generating':loaded?'Not generated':'Loading';
 return <>
  <section className="rounded-xl border border-white/15 p-4 my-4" aria-label="Note versions">
   <div className="flex flex-wrap items-center justify-between gap-3">
    <div className="flex flex-wrap gap-2" role="group" aria-label="Choose note version">{[['original','Original'],['improved','Improved'],['compare','Side by side']].map(([key,label])=><button key={key} aria-pressed={view===key} onClick={()=>setView(key)} className={`px-4 py-2 rounded-lg font-medium text-sm ${view===key?'bg-amber-600 text-white':'bg-white/5 hover:bg-white/10'}`}>{label}</button>)}</div>
    <span role="status" className="text-sm text-slate-400">Improved notes · {status}</span>
   </div>
   <p className="text-xs text-slate-400 mt-3">New summaries automatically generate both versions. Your original notes stay intact.</p>
  </section>
  {view==='original'?children:<section className="space-y-5" aria-label={view==='compare'?'Side-by-side notes':'Improved notes'}>
   {error&&<p role="alert" className="text-red-400">{error}</p>}
   {!loaded&&!error&&<p role="status">Loading improved notes…</p>}
   {loaded&&!row&&<div className="rounded-xl border border-white/15 p-5"><h3 className="font-semibold">Generate the improved version for this saved note</h3><p className="text-sm text-slate-400 my-3">Older notes need one initial run. Charlie uses the saved transcript; no audio upload is needed. This uses your research API credits.</p><button disabled={busy||!summary.rawNotes?.trim()} onClick={start} className="bg-amber-600 text-white rounded-lg px-4 py-2 disabled:opacity-50">{busy?'Starting…':'Generate improved notes'}</button>{!summary.rawNotes?.trim()&&<p className="mt-2 text-sm">No saved source text is available.</p>}</div>}
   {row&&<>
    <div className="rounded-xl border border-amber-500/30 p-4 space-y-3"><strong className="text-sm">Improved notes · All sections</strong>{controls()}<p role="status" className="text-sm text-slate-400">{exportMessage||(row.status==='complete'?'Exports use this saved improved version and leave original notes intact.':'Export controls become available when generation finishes.')}</p></div>
    {rows.length>1&&<label className="block text-sm">Saved version <select className="bg-transparent border border-white/20 p-2 rounded" value={row.id} onChange={e=>setSelected(e.target.value)}>{rows.map(r=><option key={r.id} value={r.id}>{new Date(r.created_at).toLocaleString()} · {r.status}</option>)}</select></label>}
    {running&&<div role="status" className="rounded-xl border border-amber-500/30 p-4"><strong>{state.progress||'Queued for generation'}</strong><p className="text-sm text-slate-400 mt-1">You can leave this page. Completed sections appear below as they are saved.</p>{row.recovery_enabled&&<p className="text-sm text-slate-400">Interrupted work can resume automatically with the server research key. Recovery attempts: {row.recovery_attempts||0} of 2. Provider failures still require review.</p>}</div>}
    {(row.status==='failed'||stale)&&<div role="alert" className="rounded-xl border border-amber-500/30 p-4"><p>{row.error||'No recent progress. Resume from the last saved checkpoint.'}</p><button disabled={busy} onClick={start} className="underline mt-2">{busy?'Starting…':'Retry improved notes'}</button></div>}
    {view==='compare'&&<p className="text-xs text-slate-400">Original on the left, improved on the right. The original is the saved snapshot from when this version began. On mobile, each pair is stacked.</p>}
    {sections.map(([key,label,original])=>{const text=key==='record'?record:state.sections?.[key];return <section key={key} className="space-y-3"><div className="flex flex-wrap justify-between items-center gap-3"><h3 className="text-lg font-semibold">{label}</h3>{controls(key)}</div><div className={`grid gap-4 ${view==='compare'?'xl:grid-cols-2':'grid-cols-1'}`}>
     {view==='compare'&&<article className="min-w-0 rounded-xl border border-white/10 p-5"><h4 className="text-xs uppercase tracking-wide text-slate-400 mb-4">Original</h4><div className="prose prose-invert max-w-none text-sm break-words" dangerouslySetInnerHTML={{__html:renderHtml(row.baseline?.[original]||'<p>No original section saved.</p>')}}/></article>}
     <article className="min-w-0 rounded-xl border border-amber-500/20 p-5">{view==='compare'&&<h4 className="text-xs uppercase tracking-wide text-amber-500 mb-4">Improved</h4>}<div className="whitespace-pre-wrap break-words text-sm leading-relaxed">{text||'Waiting for this section…'}</div></article>
    </div></section>;})}
    <details className="text-sm text-slate-400"><summary className="cursor-pointer">Source coverage and method</summary><p className="mt-2">{(state.coveredCharacters||0).toLocaleString()} / {(state.sourceCharacters||summary.rawNotes?.length||0).toLocaleString()} source characters processed. Full management records are retained. Interpretations are AI analysis, not independent verification or your own views. {state.hierarchicalSynthesis?'Long-source synthesis uses consolidated evidence.':''}</p></details>
    <label className="block text-sm font-medium">Your comparison feedback<textarea className="block w-full bg-transparent border border-white/20 rounded-lg p-3 mt-2" rows={3} maxLength={10000} value={feedback} onChange={e=>{setFeedback(e.target.value);setSaved(false);}} placeholder="Which version works better? What should Charlie preserve or improve?"/></label>
    <button className="underline text-sm" onClick={async()=>{try{const r=await fetch(`${base}/${row.id}/feedback`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({feedback}),signal:AbortSignal.timeout(20000)});if(!r.ok)throw Error('Feedback could not be saved.');setSaved(true);}catch(e){setError(e.message);}}}>Save feedback</button>{saved&&<span role="status" className="text-sm ml-3">Saved</span>}
   </>}
  </section>}
 </>;
}
