import {MeetingAnswerCapture} from './meeting-answer-capture';
import {originLink} from './research-meeting-model.mjs';
import * as React from 'react';
import {checkedMeetingContext} from './meeting-context-model.mjs';
function useContextRead(api,meetingId,questionSetId,part,enabled){
 const [state,setState]=React.useState({data:null,error:'',loading:false}),[attempt,setAttempt]=React.useState(0);
 const sequence=React.useRef(0);
 React.useEffect(()=>{if(!enabled)return;let active=true;const serial=++sequence.current;
  setState(previous=>({...previous,error:'',loading:true}));
  (async()=>{try{const suffix=part==='quality'?`?questionSetId=${questionSetId}`:'';
   const r=await fetch(`${api}/api/mp/meetings/${meetingId}/${part}${suffix}`,{signal:AbortSignal.timeout(20000)});
   if(!r.ok)throw Error('Saved '+part+' could not be loaded.');
   const data=checkedMeetingContext(part,await r.json(),meetingId,questionSetId);
   if(active&&serial===sequence.current)setState({data,error:'',loading:false});
  }catch(e){if(active&&serial===sequence.current)setState(previous=>({...previous,error:e.message,loading:false}));}})();
  return()=>{active=false;};
 },[api,meetingId,questionSetId,part,enabled,attempt]);
 return {...state,retry:()=>setAttempt(n=>n+1),update:fn=>{sequence.current++;setState(previous=>({...previous,data:fn(previous.data),loading:false}));}};
}
function ContextStatus({name,read}){return <div>{read.loading&&<p role="status">Loading {name}…</p>}{read.error&&<p role="alert">{name}: {read.error} {read.data?'Previously loaded records remain visible. ':''}<button disabled={read.loading} onClick={read.retry}>Retry {name}</button></p>}</div>;}
export function MeetingSession(props){return <MeetingSessionView key={props.api+'|'+props.meetingId+'|'+props.questionSet.id} {...props}/>;}
function MeetingSessionView({api,meetingId,questionSet}){
 const [open,setOpen]=React.useState(false),[high,setHigh]=React.useState(true),[history,setHistory]=React.useState(false);
 const answerRead=useContextRead(api,meetingId,questionSet.id,'answers',open);
 const versionRead=useContextRead(api,meetingId,questionSet.id,'versions',history);
 const qualityRead=useContextRead(api,meetingId,questionSet.id,'quality',open||history);
 const quality=qualityRead.data,answers=answerRead.data?.answers||[],versions=versionRead.data?.versions||[];
 return <section className="workspace-panel mb-3"><div className="desk-filter"><button onClick={()=>setOpen(!open)}>{open?'Close meeting mode':'Open meeting mode'}</button><button onClick={()=>setHistory(!history)}>{history?'Hide version history':'Compare saved versions'}</button></div>{open&&<ContextStatus name="saved answers" read={answerRead}/>} {history&&<ContextStatus name="version history" read={versionRead}/>} {(open||history)&&<ContextStatus name="quality checks" read={qualityRead}/>}{quality&&(open||history)&&<p className="desk-explainer">{quality.questions} saved items · {quality.highPriority} must-ask · {quality.withVerifiedPassages} with verified passage records · {quality.citedSources}/{quality.availableSources} sources cited · {quality.duplicateQuestions} exact duplicates. {quality.scope}</p>}{quality?.nonQuestionItems?.length>0&&(open||history)&&<p className="desk-explainer">Review items {quality.nonQuestionItems.join(', ')}: they may be coverage notes rather than management questions. Ask Charlie to revise these before using them in a meeting.</p>}{quality?.numericPremisesToReview?.length>0&&(open||history)&&<details><summary>Numeric premises to review · {quality.numericPremisesToReview.length} questions</summary><p>These values were not found verbatim in their attached passages. They may be valid periods, requested targets or calculations; check the original before treating them as established facts.</p>{quality.numericPremisesToReview.map(r=><p key={r.question}>Question {r.question}: {r.values.join(', ')}</p>)}</details>}{history&&<div><p>Previous versions remain unchanged. Review revisions before using them in a meeting.</p>{versionRead.data?.truncated&&<p>Showing the newest 30 saved versions. Older versions remain saved; this list is incomplete.</p>}{versionRead.data&&!versions.length&&<p>No saved versions were returned for this meeting.</p>}{versions.map(v=><details key={v.id}><summary>Version {v.version} · {v.topics.reduce((n,t)=>n+(t.questions||[]).length,0)} questions{v.id===questionSet.id?' · Current':''}</summary>{v.topics.map((t,i)=><div key={i}><h4>{t.topic}</h4><ol>{t.questions.map((q,j)=><li key={j}>{q.question}</li>)}</ol></div>)}</details>)}</div>}{open&&<div><h3>Meeting mode · v{questionSet.version}</h3><label><input type="checkbox" checked={high} onChange={e=>setHigh(e.target.checked)}/> Must-ask questions only</label><p className="desk-explainer">Capture actual management answers here. Saving an answer marks the question answered; generating a question does not.</p>{questionSet.topics.map((t,i)=><section key={i}><h4>{t.topic}</h4>{t.questions.filter(q=>!high||q.priority==='high').map((q,j)=><article className="workspace-panel" key={q.question}><p><strong>{q.question}</strong></p><details><summary>Evidence and follow-up</summary><p>{q.context}</p><p>{q.source}</p>{originLink(q.research_origin)&&<p><a href={originLink(q.research_origin)}>Open the saved {q.research_origin.ticker} research report</a> · {q.research_origin.path}</p>}{q.supporting_quotes?.map((c,k)=><blockquote key={k}>{c.quote}<cite> — {c.filename}</cite></blockquote>)}<p>{q.follow_up_angle}</p></details><MeetingAnswerCapture api={api} meetingId={meetingId} question={q.question} answers={answers} onSaved={row=>answerRead.update(previous=>({...previous,answers:(previous?.answers||[]).map(a=>a.id===row.id?row:a)}))}/></article>)}</section>)}</div>}</section>;
}
