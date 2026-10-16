import * as React from 'react';
import {originLink} from './research-meeting-model.mjs';
export function AssignmentMeetingVersion({api,assignmentId,meetingId,ticker,version}) {
 const [open,setOpen]=React.useState(false),[data,setData]=React.useState(null),[error,setError]=React.useState(''),[attempt,setAttempt]=React.useState(0);
 React.useEffect(()=>{
  setData(null);setError('');if(!open)return;let active=true;
  (async()=>{try{
   const r=await fetch(`${api}/api/research/assignments/${assignmentId}/meeting-versions/${version.id}`,{signal:AbortSignal.timeout(20000)});
   if(!r.ok)throw Error('This saved version could not be loaded. Retry to read the same version.');
   const d=await r.json(),q=d.questionSet;
   if(d.assignmentId!==assignmentId||q?.id!==version.id||q.meeting_id!==meetingId||q.ticker!==ticker||q.version!==version.version||!Array.isArray(q.topics))throw Error('Saved version identity could not be verified. No other version has been substituted.');
   if(active)setData(q);
  }catch(e){if(active)setError(e.message);}})();return()=>{active=false;};
 },[open,attempt,api,assignmentId,meetingId,ticker,version.id,version.version]);
 return <article style={{overflowWrap:'anywhere'}}><button aria-expanded={open} onClick={()=>setOpen(!open)}>{open?'Close':'Inspect'} saved version {version.version}</button>{open&&<section aria-label={`Saved meeting version ${version.version}`}>
 <p>Read-only saved questions and evidence. This does not generate research, accept a thesis change, or record a management answer.</p>
 {error?<p role="alert">{error} <button onClick={()=>setAttempt(n=>n+1)}>Retry saved version {version.version}</button></p>:!data?<p role="status">Loading saved version {version.version}…</p>:<>
 <h4>{ticker} · Version {data.version} · {data.status}</h4><p>{new Date(data.created_at).toLocaleString()} · {data.generation_model}</p>
 {data.topics.length===0&&<p>No questions were saved in this version.</p>}
 {data.topics.map((topic,i)=><section key={i}><h5>{topic.topic}</h5><p>{topic.description}</p><ol>{(topic.questions||[]).map((q,j)=><li key={j}><strong>{q.question}</strong>{q.priority&&<p>{q.priority} priority</p>}<p>{q.context}</p>{q.source&&<p>Source: {q.source}</p>}{q.source_filenames?.length>0&&<p>Saved source references: {q.source_filenames.join(', ')}</p>}{q.supporting_quotes?.length>0?q.supporting_quotes.map((c,k)=><blockquote key={k}><p>{c.quote}</p><cite>{c.filename}</cite></blockquote>):<p>No supporting passages were saved with this question. Check the original sources before relying on its premise.</p>}{q.source_support&&<p>{q.source_support}</p>}{originLink(q.research_origin)&&<p><a href={originLink(q.research_origin)}>Open saved research report</a></p>}{q.follow_up_angle&&<p>Follow-up: {q.follow_up_angle}</p>}</li>)}</ol></section>)}</>}
 </section>}</article>;
}
