import * as React from 'react';
import {checkedProcessingReadiness} from './processing-readiness-model.mjs';
export function ProcessingReadiness({proposal,call,path,disabled,onApprove}){
 const [data,setData]=React.useState(null),[error,setError]=React.useState(''),[busy,setBusy]=React.useState(false);
 const live=React.useRef(true),lock=React.useRef(false);
 React.useEffect(()=>{live.current=true;return()=>{live.current=false;};},[]);
 const check=async()=>{if(lock.current)return;lock.current=true;setBusy(true);setError('');try{
  const result=checkedProcessingReadiness(await call(path+'/'+encodeURIComponent(proposal.id)+'/readiness'),proposal);
  if(live.current)setData(result);
 }catch(e){if(live.current)setError(e.message);}finally{lock.current=false;if(live.current)setBusy(false);}};
 return <section aria-label="Processing readiness" style={{padding:'12px 0',overflowWrap:'anywhere'}}>
  <p>Check the saved sources and baseline before approving. This read checks original bytes, permissions, thesis/research baseline and local dispatch prerequisites. It starts no generation and spends no research credits.</p>
  <button disabled={disabled||busy} onClick={check}>{busy?'Checking saved proposal…':data?'Recheck processing readiness':'Check processing readiness · no generation'}</button>
  {error&&<p role="alert">{error} Any retained check is out of date. Recheck before approving.</p>}
  {data&&<><p role="status">{data.ready?'No blocker found at this check.':'Processing is blocked.'} Checked {new Date(data.checkedAt).toLocaleString()}.</p>
   {data.blockers.length>0&&<ul>{data.blockers.map((b,i)=><li key={i}>{b}</li>)}</ul>}
   <p><small>This is not a reservation, provider availability guarantee or approval. Approval checks the current state again; finished research still requires investor review.</small></p></>}
  <button className="workspace-primary" disabled={disabled||busy||!!error||!data?.ready} onClick={()=>{setData(null);onApprove();}}>Approve processing · uses research credits</button>
 </section>;
}
