import * as React from 'react';
import {checkedAssignmentMeeting} from './assignment-review-model.mjs';
import {originLink} from './research-meeting-model.mjs';
export function AssignmentMeetingOutput({api,job,doc}) {
 const identity=JSON.stringify([api,job.id,job.ticker,job.result.reportId,job.result.meetingId,job.result.meetingQuestionSetId]);
 return <SavedMeeting key={identity} api={api} job={job} doc={doc}/>;
}
function SavedMeeting({api,job,doc}) {
 const [data,setData]=React.useState(null),[error,setError]=React.useState(''),[attempt,retry]=React.useState(0);
 React.useEffect(()=>{let live=true;const controller=new AbortController();setData(null);setError('');
  (async()=>{try{
   const r=await fetch(`${api}/api/research/assignments/${encodeURIComponent(job.id)}/saved-meeting`,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(20000)])});
   if(!r.ok)throw Error('This assignment’s exact meeting version could not be opened. No newer version has been substituted.');
   const saved=checkedAssignmentMeeting(await r.json(),job);if(live)setData(saved);
  }catch(e){if(live)setError(e.message);}})();return()=>{live=false;controller.abort();};
 },[attempt]);
 return <section className="review-evidence" aria-label="Delivered meeting questions">
  <h3>Meeting questions from this assignment</h3><p>This is the saved version delivered from the shared research pack. The current meeting may contain later revisions. Opening either view starts no generation.</p>
  {error?<p role="alert">{error} <button onClick={()=>retry(n=>n+1)}>Retry delivered version</button></p>:!data?<p role="status">Loading the delivered meeting version…</p>:<>
   <p><strong>{job.ticker} · Delivered version {data.version}</strong> · {data.status} · {new Date(data.created_at).toLocaleString()}</p>
   {!data.topics.length&&<p>No questions were saved in this version.</p>}
   {data.topics.map((topic,i)=><section key={i}><h4>{topic.topic}</h4>{topic.description&&<p>{topic.description}</p>}<ol>{(topic.questions||[]).map((q,j)=><li key={j}><strong>{q.question}</strong>{q.priority&&<p>{q.priority} priority</p>}<p>{q.context}</p>{q.source&&<p>Source: {q.source}</p>}{q.source_filenames?.length>0&&<p>Saved source references: {q.source_filenames.join(', ')}</p>}{q.supporting_quotes?.length>0?q.supporting_quotes.map((c,k)=><blockquote key={k}><p>{c.quote}</p><cite>{c.filename}</cite></blockquote>):<p>No supporting passages were saved with this question. Check the original sources before relying on its premise.</p>}{q.source_support&&<p>{q.source_support}</p>}{originLink(q.research_origin)&&<p><a style={{color:"inherit"}} href={originLink(q.research_origin)}>Open saved research report</a></p>}{q.follow_up_angle&&<p>Follow-up: {q.follow_up_angle}</p>}</li>)}</ol></section>)}
  </>}
  <p><a className="review-link" href={doc.href}>Open current meeting workspace</a></p><p>Use the current workspace to record actual answers and follow-up. This saved view does not accept research or change the thesis.</p>
 </section>;
}
