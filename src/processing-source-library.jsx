import * as React from 'react';
import {labDocuments,visibleLabDocuments,groupLabDocuments,labSourceDate,taskSourceGuide} from './pipeline-lab-model.mjs';

export function ProcessingSourceLibrary({call,path,selected,onChange,purpose,remaining}){
 const [documents,setDocuments]=React.useState([]),[loaded,setLoaded]=React.useState(false),[busy,setBusy]=React.useState(false),[error,setError]=React.useState(''),[cursor,setCursor]=React.useState(null),[query,setQuery]=React.useState(''),[type,setType]=React.useState('All');
 const lock=React.useRef(false),live=React.useRef(true);
 React.useEffect(()=>{live.current=true;return()=>{live.current=false;};},[]);
 const load=async(after=null)=>{if(lock.current)return;lock.current=true;setBusy(true);setError('');try{
  const data=await call(path+(after?'?after='+encodeURIComponent(after):''));
  if(!Array.isArray(data.documents)||data.documents.some(r=>!/^thesis_[1-9][0-9]{0,18}$/.test(r.id)||typeof r.eligible!=='boolean')||(data.nextCursor!==null&&!/^[1-9][0-9]{0,18}$/.test(data.nextCursor)))throw Error('Source inventory response is incomplete. Reload the library.');
  const merged=after?[...documents,...data.documents]:data.documents;
  labDocuments(merged,[]); // Reject ambiguous identities before showing choices.
  if(live.current){setDocuments(merged);setCursor(data.nextCursor);setLoaded(true);}
 }catch(e){if(live.current)setError(e.message);}finally{lock.current=false;if(live.current)setBusy(false);}};
 const rows=labDocuments(documents,[]),shown=visibleLabDocuments(rows,query,type,'date');
 const missing=selected.filter(id=>!documents.some(d=>d.id==='thesis_'+id));
 return <section aria-label="Supplemental source library" style={{margin:'16px 0',padding:12,border:'1px solid var(--border-color,#d8dedd)'}}>
  <strong>{purpose==='event_note'?'Add historical context for this event':'Add foundational and historical evidence'}</strong>
  <p>Stored originals for this company outside this collection. Older sources remain eligible. Metadata suggests relevance; review the original before selecting. Files are checked byte-for-byte when the proposal is saved and approved.</p>
  {!loaded?<button disabled={busy} onClick={()=>load()}>Browse wider source library</button>:<>
   <p>{selected.length} supplemental originals selected · {documents.length} library entries loaded{cursor?' · more available':''}. No library documents are selected automatically.</p>
   <div className="sa-form-grid"><label>Find library sources<input aria-label="Find library sources" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Filename, publisher or author"/></label><label>Source type<select aria-label="Library source type" value={type} onChange={e=>setType(e.target.value)}>{['All',...new Set(rows.map(r=>r.type))].map(t=><option key={t}>{t}</option>)}</select></label></div>
   <p><small>Newest recorded publication dates first; unknown dates last. Filtering covers loaded entries only. A filename-based type suggestion is labeled below.</small></p>
   <div style={{maxHeight:360,overflow:'auto'}}>{groupLabDocuments(shown).map(group=><section key={group.label}><h5>{group.label}</h5><p><small>{taskSourceGuide(group.label,purpose==='event_note'?'note':'thesis')}</small></p>{group.rows.map(r=>{
    const id=r.id.slice(7),checked=selected.includes(id);
    return <label className="sa-check" key={r.id} style={{display:'flex',alignItems:'start',gap:8,padding:'8px 0'}}><input type="checkbox" checked={checked} disabled={!r.eligible||(!checked&&selected.length>=remaining)} onChange={e=>onChange(e.target.checked?[...selected,id]:selected.filter(v=>v!==id))}/><span>{r.filename}<small style={{display:'block'}}>Published: {labSourceDate(r.publishedDate)}{r.publisher?' · '+r.publisher:''} · {r.typeRecorded?'Recorded type':'Suggested type from filename'}{!r.eligible?' · Excluded or reference-only':''}</small></span></label>;
   })}</section>)}{!shown.length&&<p>No matching entries in the loaded library.</p>}</div>
   {cursor&&<button disabled={busy} onClick={()=>load(cursor)}>Load more library sources</button>}
  </>}
  {busy&&<p role="status">Loading source metadata…</p>}{error&&<p role="alert">{error} <button disabled={busy} onClick={()=>load(loaded?cursor:null)}>Retry library read</button></p>}
  {missing.length>0&&<p role="alert">{missing.length} selected identities are outside the currently loaded entries. Review them before saving. <button onClick={()=>onChange(selected.filter(id=>!missing.includes(id)))}>Remove unseen selections</button></p>}
 </section>;
}
