import React from 'react';

export default function LabICloudImport({api,ticker,task,revision,selected,disabled,onImported,read}) {
 const [jobs,setJobs]=React.useState([]),[ready,setReady]=React.useState(false),[error,setError]=React.useState(''),[busy,setBusy]=React.useState(false),[pending,setPending]=React.useState(false);
 const alive=React.useRef(true),lock=React.useRef(false),seen=React.useRef(null),callback=React.useRef(onImported);callback.current=onImported;
 const key=`charlie-lab-icloud-v1:${api}:${ticker}`;
 const refresh=async()=>{try{const d=await read(api,`/api/pipeline-lab/icloud-imports/${ticker}`);if(!Array.isArray(d.imports))throw Error('Import status unavailable.');if(!alive.current)return;
  const completed=d.imports.filter(j=>j.status==='complete').map(j=>j.requestId);
  const changed=seen.current&&completed.some(id=>!seen.current.has(id));seen.current=new Set(completed);setJobs(d.imports);setReady(true);if(changed)callback.current();
 }catch(e){if(alive.current){setError(e.message);setReady(false);}}};
 React.useEffect(()=>{alive.current=true;try{setPending(!!sessionStorage.getItem(key));}catch{setError('Retry storage unavailable; imports cannot start safely.');}refresh();const timer=setInterval(refresh,10000);return()=>{alive.current=false;clearInterval(timer);};},[api,ticker]);
 const run=async(action,job)=>{if(lock.current)return;lock.current=true;setBusy(true);setError('');try{
  let result;
  if(action==='start'){
   const prior=sessionStorage.getItem(key);const body=prior?JSON.parse(prior):{requestId:crypto.randomUUID(),task,revision};
   if(!prior)sessionStorage.setItem(key,JSON.stringify(body));if(sessionStorage.getItem(key)!==JSON.stringify(body))throw Error('Could not preserve request identity. Nothing was submitted.');setPending(true);
   result=await read(api,`/api/pipeline-lab/icloud-imports/${ticker}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
   if(result.requestId!==body.requestId)throw Error('Import receipt unavailable. Reconcile the same request.');sessionStorage.removeItem(key);setPending(false);
  }else result=await read(api,`/api/pipeline-lab/icloud-import/${job.requestId}/${action}`,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  if(result.status==='complete')callback.current();await refresh();
 }catch(e){if(action==='start'&&[400,409].includes(e.status)){try{const old=JSON.parse(sessionStorage.getItem(key));await read(api,`/api/pipeline-lab/icloud-import/${old.requestId}`);}catch(check){if(check.status===404){sessionStorage.removeItem(key);if(alive.current)setPending(false);}}}if(alive.current)setError(e.message+' Existing originals remain preserved.');}finally{lock.current=false;if(alive.current)setBusy(false);}};
 const active=jobs.find(j=>['queued','frozen','attention'].includes(j.status));
 return <section style={{marginTop:20}} aria-label="iCloud original imports"><h3>Import selected iCloud originals</h3><p>Copies the exact selected folders and files into Charlie, then updates only this task’s selection. Originals stay in iCloud. Requires the Mac online, its agent running and files downloaded. No research starts.</p>
 {error&&<p role="alert">{error}</p>}
 {(selected.some(r=>r.startsWith('icloud:'))||pending)&&<button disabled={disabled||busy||!ready||(!pending&&!!active)} onClick={()=>run('start')}>{pending?'Reconcile same iCloud request':'Import selected iCloud originals'}</button>}
 {active&&<div role="status"><p>Import {active.status} · {active.input.task} sources · saved revision {active.input.revision}</p>{active.issue&&<p>{active.issue}</p>}<p><small>{active.status==='queued'?'Waiting for the Mac agent. This does not confirm the files are available.':active.status==='frozen'?'Original hashes recorded; waiting for verified cloud receipts.':'Resolve the issue, then retry the same frozen originals or cancel and save a new selection.'}</small></p>{active.status==='attention'&&<button disabled={busy||disabled} onClick={()=>run('retry',active)}>Retry this import</button>}<button disabled={busy} onClick={()=>run('cancel',active)}>Cancel this import</button></div>}
 {jobs.filter(j=>j.status==='complete').slice(0,1).map(j=><p key={j.requestId}>Last completed import: {j.receipts.length} original(s) verified · {j.input.task} selection updated to revision {j.completedRevision}. No research was started.</p>)}
 <button disabled={busy} onClick={refresh}>Refresh import status</button></section>;
}
