import * as React from 'react';
import {researchAlerts,incompleteResearchAlerts} from './research-alert-model.mjs';
import {activityDestination} from './activity-model.mjs';
export function ResearchAlerts({api,onNavigate}) {
 const [data,setData]=React.useState(null),[dismissed,setDismissed]=React.useState({}),[unavailable,setUnavailable]=React.useState(false);
 React.useEffect(()=>{let alive=true,pending=false;const controller=new AbortController();
  const load=async()=>{if(pending)return;pending=true;try{const r=await fetch(api+'/api/activity',{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(15000)])});if(!r.ok)throw Error();const d=await r.json();if(alive){setData(d);setUnavailable(incompleteResearchAlerts(d));}}catch{if(alive)setUnavailable(true);}finally{pending=false;}};
  load();const timer=setInterval(load,30000);const visible=()=>{if(!document.hidden)load();};document.addEventListener('visibilitychange',visible);
  return()=>{alive=false;controller.abort();clearInterval(timer);document.removeEventListener('visibilitychange',visible);};},[api]);
 const alerts=researchAlerts(data,dismissed),row=alerts[0];
 if(!row&&!unavailable)return null;
 return <ResearchAlertBanner row={row} count={alerts.length} unavailable={unavailable} onNavigate={onNavigate} onDismiss={()=>setDismissed(d=>({...d,[row.alertKey]:true}))}/>;
}
export function ResearchAlertBanner({row,count=1,unavailable=false,onNavigate,onDismiss}) {
 return <aside role="alert" className="research-global-alert"><style>{`.research-global-alert{flex-shrink:0;background:#fff2da;color:#171510;border-bottom:1px solid #d3b77d;padding:12px 24px;display:flex;align-items:center;gap:16px;font:15px/1.4 Calibri,sans-serif}.research-global-alert p{margin:0;flex:1;min-width:0;overflow-wrap:anywhere}.research-global-alert button{font:inherit;color:#171510;background:#fff;border:1px solid #a58d61;border-radius:5px;padding:7px 12px;flex-shrink:0}@media(max-width:650px){.research-global-alert{padding:10px 14px;flex-wrap:wrap;gap:8px}.research-global-alert p{flex-basis:100%}}`}</style>
 <p>{row?<><strong>{row.ticker} research needs attention.</strong> {row.error||row.step||'Open the saved issue before retrying.'}{count>1&&` ${count-1} other companies also need attention.`}</>:<strong>Research alerts may be incomplete.</strong>} {unavailable&&'Some statuses may be missing or out of date. Check Activity and the company workspace.'}</p>
 <button onClick={()=>{if(row){const dest=activityDestination(row);onNavigate(dest.view,row.ticker,dest.section,dest.work);}else onNavigate('activity');}}>{row?'Open saved work':'Open Activity'}</button>
 {row&&<button aria-label={`Dismiss ${row.ticker} research alert for this session`} onClick={onDismiss}>Dismiss</button>}
 </aside>;
}
