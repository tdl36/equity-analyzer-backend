import * as React from 'react';
import {collectedSources,duplicateSelections} from './collected-source-model.mjs';
import {visibleLabDocuments,groupLabDocuments,labSourceDate,taskSourceGuide} from './pipeline-lab-model.mjs';
export function CollectedSourcePicker({api,call,job,selected,onChange,purpose,remaining,onReady}){
 const [rows,setRows]=React.useState([]),[error,setError]=React.useState(''),[busy,setBusy]=React.useState(false),[loaded,setLoaded]=React.useState(false),[query,setQuery]=React.useState(''),[type,setType]=React.useState('All'),[onlySelected,setOnlySelected]=React.useState(false);
 const live=React.useRef(false),lock=React.useRef(false);
 const load=async()=>{if(lock.current)return;lock.current=true;setBusy(true);onReady(false);try{
  const result=collectedSources(await call('/api/research/assignments/'+job.id+'/processing-sources'),job);
  if(live.current){setRows(result);setLoaded(true);setError('');onReady(true);}
 }catch(e){if(live.current)setError(e.message);}finally{lock.current=false;if(live.current)setBusy(false);}};
 React.useEffect(()=>{live.current=true;load();return()=>{live.current=false;};},[job.id,api]);
 const duplicates=duplicateSelections(rows,selected),shown=visibleLabDocuments(rows.filter(r=>!onlySelected||selected.includes(r.filename)),query,type,'date');
 const hidden=selected.filter(name=>!shown.some(r=>r.filename===name)).length;
 return <section aria-label="Collected evidence" style={{margin:'16px 0',padding:12,border:'1px solid var(--border-color,#d8dedd)',overflowWrap:'anywhere'}}>
  <strong>Choose collected evidence</strong>
  <p>{selected.length} selected from {job.result.sources.length} collection receipts · no sources selected automatically.</p>
  <p><small>Newest recorded publication dates first; unknown dates last. Metadata helps organize evidence and is not a content review. Stored-file links open the current Charlie original. Its bytes and permissions are checked against the collection receipt when saving and approving.</small></p>
  <div className="sa-form-grid"><label>Find collected sources<input aria-label="Find collected sources" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Filename, publisher or author"/></label><label>Collected source type<select aria-label="Collected source type" value={type} onChange={e=>setType(e.target.value)}>{['All',...new Set(rows.map(r=>r.type))].map(t=><option key={t}>{t}</option>)}</select></label></div>
  <label className="sa-check"><input type="checkbox" checked={onlySelected} onChange={e=>setOnlySelected(e.target.checked)}/>Show selected collected sources only</label>
  {hidden>0&&<p role="status">{hidden} selections hidden by these filters. Filtering never changes your selections.</p>}
  <div style={{maxHeight:380,overflow:'auto'}}>{groupLabDocuments(shown).map(group=><section key={group.label}><h5>{group.label}</h5><p><small>{taskSourceGuide(group.label,purpose==='event_note'?'note':'thesis')}</small></p>{group.rows.map(r=>{
   const checked=selected.includes(r.filename),duplicate=duplicates.has(r.filename);
   return <div key={r.filename} style={{padding:'10px 0',borderTop:'1px solid var(--border-color,#d8dedd)'}}><label style={{display:'flex',alignItems:'start',gap:8}}><input type="checkbox" aria-label={r.filename} checked={checked} disabled={!checked&&(busy||!!error||r.availability!=='available'||duplicate||selected.length>=remaining)} onChange={e=>onChange(e.target.checked?[...selected,r.filename]:selected.filter(n=>n!==r.filename))}/><span>{r.filename}</span></label><small style={{display:'block',marginLeft:24}}>Published: {labSourceDate(r.publishedDate)}{r.publisher?' · '+r.publisher:''}{r.author?' · '+r.author:''} · {r.typeRecorded?'Recorded document type':'Type suggested from filename'}{r.availability!=='available'?' · '+({missing:'Stored original missing',ambiguous:'Stored identity ambiguous',restricted:'Excluded or reference-only'}[r.availability]):''}{duplicate?' · Same receipt bytes as a selected source; keep one copy':''}</small>{r.id&&r.availability==='available'&&<a style={{marginLeft:24}} href={api+'/api/pipeline-lab/original/'+encodeURIComponent(job.ticker)+'/'+r.id} target="_blank" rel="noreferrer">Open stored original</a>}</div>;
  })}</section>)}{loaded&&!shown.length&&<p>No collected sources match these filters.</p>}</div>
  {busy&&<p role="status">Loading collected source metadata…</p>}{error&&<p role="alert">{error} Your saved selections are retained.</p>}
  <button disabled={busy} onClick={load}>Refresh collected source metadata</button>
 </section>;
}
