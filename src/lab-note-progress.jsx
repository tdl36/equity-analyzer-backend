import * as React from 'react';
import {checkedNoteProgress,batchOutcome} from './lab-note-progress-model.mjs';
export function LabNoteProgress(props){return <Inspection key={JSON.stringify([props.api,props.ticker,props.id])} {...props}/>;}
function Inspection({api,ticker,id,read,onOpen,onRecover,disabled,describeReason}) {
 const [data,setData]=React.useState(null),[error,setError]=React.useState(''),[busy,setBusy]=React.useState(false),[attempt,refresh]=React.useState(0);
 React.useEffect(()=>{let live=true,pending=false;const controller=new AbortController();
  const load=async()=>{if(pending)return;pending=true;setBusy(true);try{
   const d=checkedNoteProgress(await read(api,`/api/pipeline-lab/note-job/${encodeURIComponent(id)}/inspection?ticker=${encodeURIComponent(ticker)}`,{signal:controller.signal}),id,ticker);
   if(live){setData(d);setError('');}
  }catch(e){if(live)setError('Saved progress could not be verified. Retained information may be out of date; retry this read before recovery.');}finally{pending=false;if(live)setBusy(false);}};
  load();const timer=setInterval(load,10000);return()=>{live=false;controller.abort();clearInterval(timer);};
 },[api,ticker,id,attempt]);
 return <section aria-label="Exact full-note progress" style={{border:'1px solid #8885',padding:16,margin:'16px 0',overflowWrap:'anywhere'}}>
  <h4>{ticker} · Selected full-note progress</h4><p><small>Saved note {id}</small></p>
  <button disabled={busy} onClick={()=>refresh(n=>n+1)}>Refresh selected progress</button>
  {busy&&!data&&<p role="status">Loading this note’s saved progress…</p>}{error&&<p role="alert">{error}</p>}
  {data&&<><p><strong>{data.status==='draft'?'Draft saved':data.status}</strong> · {data.modelKey} · updated {new Date(data.updatedAt).toLocaleString()}</p>
   {data.reason&&<p>{describeReason?.(data.reason)||'Inspect the saved outcome before continuing.'}</p>}
   <ol>{Array.from({length:data.batchCount},(_,i)=>{const call=data.batches.find(b=>b.batch===i+1);return <li key={i}>Batch {i+1}: {batchOutcome(call?.status)}</li>;})}</ol>
   <p>A dispatch record does not confirm a charge or provider completion. Saved replies do not establish an accepted research note. This read does not start or resume generation.</p>
   <div className="workspace-actions">{data.revision?<button disabled={disabled||busy||!!error} onClick={()=>onOpen(id,data.revision)}>Open selected draft revision {data.revision}</button>:<p>No draft revision is saved for this job.</p>}{data.acceptedRevision&&<button disabled={disabled||busy||!!error} onClick={()=>onOpen(id,data.acceptedRevision)}>Open selected accepted revision {data.acceptedRevision}</button>}
   {data.status==='attention'&&data.reason!=='parent_stopped'&&<button disabled={disabled||busy||!!error} onClick={async()=>{await onRecover(id);refresh(n=>n+1);}}>Reconcile selected saved progress</button>}</div>
   {data.status==='attention'&&data.reason!=='parent_stopped'&&<p>Reconciliation reuses saved replies where possible. Safe unfinished batches may resume under the original authorization and spend credits; uncertain calls are not repeated.</p>}
  </>}
 </section>;
}
