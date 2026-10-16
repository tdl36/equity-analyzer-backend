import * as React from 'react';
import {AssignmentMeetingVersion} from './assignment-meeting-version';
export function AssignmentWorkspace(props) {
 return <AssignmentInspection key={props.api+'|'+props.id} {...props}/>;
}
function AssignmentInspection({api,id,onNavigate,onSection}) {
 const [open,setOpen]=React.useState(false),[data,setData]=React.useState(null),[error,setError]=React.useState(''),[loading,setLoading]=React.useState(false),[revision,setRevision]=React.useState(0);
 React.useEffect(()=>{
  if(!open)return;let active=true,pending=false;const controller=new AbortController();
  const load=async()=>{
   if(pending)return;pending=true;setLoading(true);
   try{
    const r=await fetch(api+'/api/research/assignments/'+id,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(20000)])});
    const d=await r.json();if(!r.ok)throw Error('This assignment could not be refreshed. Retained information may be out of date.');
    if(d.command?.id!==id||typeof d.command.ticker!=='string'||!d.assignment||!Array.isArray(d.documents)||!Array.isArray(d.versions))throw Error('The saved assignment identity could not be verified. No other assignment has been substituted.');
    if(active){setData(d);setError('');}
   }catch(e){if(active)setError(e.message);}finally{pending=false;if(active)setLoading(false);}
  };
  load();const timer=setInterval(load,10000);return()=>{active=false;controller.abort();clearInterval(timer);};
 },[open,id,api,revision]);
 return <div className="assignment-workspace"><button onClick={()=>setOpen(!open)}>{open?'Close assignment workspace':'Inspect assignment →'}</button>{open&&<section aria-label="Assignment workspace"><button disabled={loading} onClick={()=>setRevision(n=>n+1)}>{loading?'Refreshing…':'Refresh assignment'}</button>{loading&&!data&&<p role="status">Loading this assignment…</p>}{error&&<p role="alert">{error}</p>}{data&&<><h3>{data.command.ticker} · Assignment workspace</h3><details className="assignment-brief"><summary>Full assignment and focus areas</summary><p>{data.assignment.instruction}</p></details><p>{data.assignment.since} through {data.assignment.until}</p>{data.delivery&&<section aria-label="Assignment delivery"><h4>{data.delivery.state==='delivered'?'Outputs delivered':data.delivery.state==='attention'?'Needs attention':data.delivery.state==='unknown'?'Status needs verification':'Work in progress'} · {data.delivery.delivered}/{data.delivery.expected} stages delivered</h4><ol>{data.delivery.steps.map(step=><li key={step.key}><strong>{step.label} · {step.state}</strong><p>{step.detail}</p></li>)}</ol><p>Delivery means saved in Charlie. It does not mean emailed, approved, or applied to your thesis.</p></section>}{(data.collection?.issue||data.preparation?.error||data.command.error)&&<p role="alert">{data.collection?.issue||data.preparation?.error||data.command.error}</p>}
 {data.meetingIssue&&<p role="alert">{data.meetingIssue}</p>}
 {data.collection?.historicalReceipt&&<p role="status">Historical collector receipt: {data.collection.status}. This records collection status only; original-byte verification and processing approval remain separate.</p>}
 {data.collection?.sourceProgress&&<details open><summary>Collection progress</summary><p>{(data.collection.sourceProgress.documents.handed_off||0)+(data.collection.sourceProgress.documents.duplicate||0)} originals saved to iCloud · {data.collection.sourceProgress.documents.staged||0} awaiting handoff · {data.collection.sourceProgress.documents.held||0} held</p>{data.collection.sourceProgress.tasks.map(t=><p key={t.kind}>{t.kind.replaceAll('-',' ')} · {t.status.replaceAll('_',' ')}{t.expected!=null?` · ${t.expected} selected`:''}</p>)}<p>Meeting originals appear below after collection and the analyst brief finish.</p></details>}
 {data.selection&&<details><summary>Selected sources and decisions</summary>{(data.selection.candidates||[]).map(c=><p key={c.url}><strong>{c.decision}</strong> · {c.publisher} · {c.title}<br/>{c.reason}</p>)}</details>}
 <details><summary>{data.documents.length} meeting originals · {data.preparation?.cachedSources||0} analyses reused</summary>{data.documents.map(d=><p key={d.filename}>{d.filename}</p>)}</details>
 <details><summary>{data.versions.length} saved meeting versions</summary>{data.versions.map(v=><div key={v.id}><p>Version {v.version} · {v.status} · {new Date(v.created_at).toLocaleString()} · {v.generation_model}</p><AssignmentMeetingVersion api={api} assignmentId={data.command.id} meetingId={data.meetingId} ticker={data.command.ticker} version={v}/></div>)}</details>
 {data.assignment.sourcePolicy&&<details><summary>Frozen source preferences</summary><p>Mode: {data.assignment.sourcePolicy.mode}</p>{(data.assignment.sourcePolicy.rules||[]).map((r,i)=><p key={i}>{r.name} · {r.analyst||'all analysts'} · {r.ticker||r.subsector||'all companies'} · {r.disposition}</p>)}</details>}
 {data.meetingId&&<button onClick={()=>{onNavigate('meetingprep',data.command.ticker,'','meeting:'+data.meetingId);}}>Open questions and supporting passages →</button>}<button onClick={()=>onSection('collection')}>Collection controls</button><button onClick={()=>onNavigate('analysts')}>Open analyst inbox</button></>}</section>}</div>;
}
