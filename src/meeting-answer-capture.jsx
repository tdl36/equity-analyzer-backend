import * as React from 'react';
import {checkedAnswer,answerPayload,answerReceipt} from './meeting-answer-capture-model.mjs';
export function MeetingAnswerCapture({api,meetingId,question,answers,onSaved}){
 const matches=answers.filter(a=>a.question===question&&a.meeting_id===meetingId);
 const [selected,setSelected]=React.useState('');
 const answer=matches.length===1?matches[0]:matches.find(a=>String(a.id)===selected);
 return <div>{matches.length>1&&<label>Several saved records use this question. Choose the record to inspect; these records do not identify a question-pack version.<select value={selected} onChange={e=>setSelected(e.target.value)}><option value="">Choose a saved answer record</option>{matches.map(a=><option key={a.id} value={a.id}>Record {a.id} · {a.status}{a.response_notes?' · has saved notes':''}</option>)}</select></label>}{answer?<AnswerEditor key={api+'|'+meetingId+'|'+answer.id} api={api} initial={answer} onSaved={onSaved}/>:<p className="desk-explainer">{matches.length?'Select the exact saved record above before recording an answer.':'Answer record unavailable for this question.'}</p>}</div>;
}
export function AnswerEditor({api,initial,onSaved}){
 const key='charlie-meeting-answer-v1:'+JSON.stringify([api,initial.meeting_id,initial.id]);
 const [base,setBase]=React.useState(initial),[text,setText]=React.useState(initial.response_notes||''),[pending,setPending]=React.useState(null),[remote,setRemote]=React.useState(null),[message,setMessage]=React.useState(''),[error,setError]=React.useState(''),[busy,setBusy]=React.useState(false),[ready,setReady]=React.useState(false);
 const alive=React.useRef(false),lock=React.useRef(false);
 const persist=(b,t,p)=>{const raw=JSON.stringify({base:b,text:t,pending:p});sessionStorage.setItem(key,raw);if(sessionStorage.getItem(key)!==raw)throw Error('Browser recovery could not be saved. No answer was sent.');};
 React.useEffect(()=>{alive.current=true;try{const raw=sessionStorage.getItem(key);if(raw){const d=JSON.parse(raw);checkedAnswer(d.base,initial.meeting_id,initial.id,initial.question);if(typeof d.text!=='string'||d.text.length>10000||d.pending!==null&&JSON.stringify(d.pending)!==JSON.stringify(answerPayload(d.base,d.text)))throw Error('Saved answer recovery is inconsistent.');setBase(d.base);setText(d.text);setPending(d.pending);setMessage('Recovered your local answer draft. Check the saved answer before changing its baseline.');}setReady(true);}catch(e){setError(e.message+' Local recovery must be resolved before saving.');}return()=>{alive.current=false;};},[key]);
 const dirty=text!==(base.response_notes||'')||!!pending;
 React.useEffect(()=>{const warn=e=>{if(dirty){e.preventDefault();e.returnValue='';}};window.addEventListener('beforeunload',warn);return()=>window.removeEventListener('beforeunload',warn);},[dirty]);
 const confirm=row=>{sessionStorage.removeItem(key);if(sessionStorage.getItem(key)!==null)throw Error('Answer saved, but local recovery could not be cleared. Check the saved answer again.');setBase(row);onSaved?.(row);setPending(null);setRemote(null);setError('');setMessage('Exact answer saved to meeting history. Research and thesis approval remain separate.');};
 const save=async()=>{if(lock.current||!ready||remote)return;lock.current=true;setBusy(true);setError('');const payload=pending||answerPayload(base,text);
  try{persist(base,text,payload);setPending(payload);const r=await fetch(`${api}/api/mp/past-questions/${base.id}/note`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:AbortSignal.timeout(20000)});const d=await r.json();if(!alive.current)return;
   if(r.status===409){setRemote(checkedAnswer(d.answer,base.meeting_id,base.id,base.question));throw Error('Another saved answer differs from your baseline. Your draft is retained; review it below.');}
   if(!r.ok)throw Error(d.error||'Save not confirmed. Check the saved answer before retrying.');confirm(answerReceipt(d,base,payload));
  }catch(e){if(alive.current)setError(e.message);}finally{lock.current=false;if(alive.current)setBusy(false);}
 };
 const check=async()=>{if(lock.current)return;lock.current=true;setBusy(true);setError('');try{const r=await fetch(`${api}/api/mp/meetings/${base.meeting_id}/answers/${base.id}`,{signal:AbortSignal.timeout(20000)});if(!r.ok)throw Error('Saved answer could not be read. Your draft is retained.');const d=await r.json();const row=checkedAnswer(d.answer,base.meeting_id,base.id,base.question);if(!alive.current)return;
  if(pending&&row.response_notes===pending.responseNotes&&row.status===pending.status)confirm(row);
  else if(row.response_notes!==base.response_notes||row.status!==base.status){setRemote(row);setMessage('Review the current saved answer below before replacing it.');}
  else setMessage(pending?'No matching saved answer found. Retry keeps the same text and baseline.':'Saved answer still matches your baseline.');
 }catch(e){if(alive.current)setError(e.message);}finally{lock.current=false;if(alive.current)setBusy(false);}};
 return <div style={{overflowWrap:'anywhere'}}><p>Saved record {base.id} · {base.status}</p><label>Management answer<textarea className="w-full" value={text} maxLength={10000} disabled={busy||!!pending||!ready} onChange={e=>{const t=e.target.value;setText(t);setMessage('Draft retained in this browser session.');try{persist(base,t,null);setError('');}catch(x){setError(x.message);}}} placeholder="Capture what management actually said…"/></label>
 <div className="workspace-actions"><button disabled={busy||!ready||!text.trim()||!!remote} onClick={save}>{busy?'Checking…':pending?'Retry same answer save':'Save answer'}</button><button disabled={busy||!ready} onClick={check}>Check saved answer</button></div>
 {remote&&<section aria-label="Changed saved answer"><h4>Current saved answer · {remote.status}</h4><p style={{whiteSpace:'pre-wrap'}}>{remote.response_notes||'No answer text saved.'}</p><p>Your draft above is unchanged. Review both before saving a replacement.</p><button disabled={busy} onClick={()=>{try{persist(remote,text,null);setBase(remote);setPending(null);setRemote(null);setError('');setMessage('Reviewed baseline selected. Your draft is unchanged; Save answer explicitly replaces that baseline.');}catch(e){setError(e.message);}}}>Use this reviewed answer as baseline</button></section>}
 {error&&<p role="alert">{error}</p>}<p role="status">{message}</p></div>;
}
