import * as React from 'react';

// Consent belongs to this saved assignment state, never to the surrounding list.
export function AssignmentResume({job,disabled,onResume}){
 const [acknowledged,setAcknowledged]=React.useState(null);
 const revision=typeof job.resumeRevision==='string'&&/^[a-f0-9]{64}$/.test(job.resumeRevision)?job.resumeRevision:null;
 const scope=revision?job.id+':'+revision:null;
 const checked=!!scope&&acknowledged===scope;
 React.useEffect(()=>{if(disabled)setAcknowledged(null);},[disabled]);
 return <section aria-label="Assignment resume controls">
  {!revision&&<p role="status">Retry controls need a current saved-state receipt. Wait for assignment status to refresh.</p>}
  <label className="sa-check"><input type="checkbox" disabled={disabled||!revision} checked={checked} onChange={e=>setAcknowledged(e.target.checked?scope:null)}/>If a paid call is uncertain, I checked usage and accept that retry may charge again.</label>
  <p><small>This acknowledgment applies only to this assignment’s current saved state and is cleared after each attempt.</small></p>
  <button disabled={disabled||!revision} onClick={()=>{setAcknowledged(null);onResume({acknowledgeRetry:checked,expectedRevision:revision});}}>Resume assignment</button>
 </section>;
}
