import * as React from 'react';
import {noteArchivePage,appendNoteArchive} from './lab-note-archive-model.mjs';
export function LabNoteArchive({api,ticker,read,onOpen,onInspect,disabled,describeReason}) {
 const [open,setOpen]=React.useState(false),[state,setState]=React.useState({rows:[],cursor:null,loaded:false,busy:false,error:''});
 const serial=React.useRef(0),locked=React.useRef(false);
 const load=async(more=false)=>{
  if(locked.current)return;locked.current=true;const token=++serial.current;
  setState(s=>({...s,busy:true,error:''}));
  try{
   const page=noteArchivePage(await read(api,`/api/pipeline-lab/note-jobs/${ticker}${more&&state.cursor?'?cursor='+encodeURIComponent(state.cursor):''}`),ticker);
   if(token===serial.current)setState(s=>({rows:more?appendNoteArchive(s.rows,page.rows):page.rows,cursor:page.cursor,loaded:true,busy:false,error:''}));
  }catch(e){if(token===serial.current)setState(s=>({...s,busy:false,error:'Saved note archive could not be loaded. Your loaded entries remain available; retry this read.'}));}
  finally{if(token===serial.current)locked.current=false;}
 };
 React.useEffect(()=>{if(open)load();return()=>{serial.current++;locked.current=false;};},[open,api,ticker]);
 return <section aria-label="Saved full-note archive" style={{margin:'20px 0',overflowWrap:'anywhere'}}>
  <button aria-expanded={open} onClick={()=>setOpen(v=>!v)}>{open?'Close full-note archive':'Find older full notes'}</button>
  {open&&<><h4>{ticker} · Saved full-note archive</h4><p>Browse all saved dates. Status reflects the last archive read. Opening a revision starts no generation and does not accept research.</p>
   <button disabled={state.busy} onClick={()=>load()}>Refresh archive from newest</button>
   {state.busy&&<p role="status">Loading saved notes…</p>}{state.error&&<p role="alert">{state.error}</p>}
   {state.loaded&&!state.rows.length&&<p>No saved full-note jobs for this company.</p>}
   {state.rows.map(row=><article key={row.noteId} style={{borderTop:'1px solid #8884',padding:'12px 0'}}><strong>{row.title}</strong><p>{new Date(row.createdAt).toLocaleString()} · {row.status==='draft'?'Draft saved':row.status}</p>
    {row.reason&&<p>{describeReason?.(row.reason)||'Inspect saved progress before retrying; an uncertain paid outcome must not be repeated.'}</p>}
    {row.acceptedRevision&&<p>Investor accepted revision {row.acceptedRevision}{row.revision>row.acceptedRevision?' · newer edits need review':''}</p>}
    <div className="workspace-actions"><button onClick={()=>onInspect(row.noteId)}>Inspect this note’s progress</button>{row.revision&&<button disabled={disabled} onClick={()=>onOpen(row.noteId,row.revision)}>Open saved revision {row.revision}</button>}{row.acceptedRevision&&row.acceptedRevision!==row.revision&&<button disabled={disabled} onClick={()=>onOpen(row.noteId,row.acceptedRevision)}>Open accepted revision {row.acceptedRevision}</button>}</div>
    {!row.revision&&<p>No saved draft revision is available for this job.</p>}
   </article>)}
   {state.cursor&&<button disabled={state.busy} onClick={()=>load(true)}>{state.error?'Retry older full notes':'Load older full notes'}</button>}
   {state.error&&!state.cursor&&<button disabled={state.busy} onClick={()=>load()}>Retry note archive</button>}
  </>}
 </section>;
}
