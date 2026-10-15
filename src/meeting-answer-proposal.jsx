import {linkedProposals,proposalLink,proposalIntentKey,readProposalIntent,reserveProposalIntent,proposalReceipt,clearProposalIntent} from './meeting-answer-proposal-model.mjs';
import * as React from 'react';
export function MeetingAnswerProposal(props) {return <ProposalHandoff key={props.api+'|'+props.ticker+'|'+props.record.id} {...props}/>;}
function ProposalHandoff({api,ticker,record,disabled=false}) {
 const [data,setData]=React.useState({jobs:[],activeProposal:null,truncated:false}),[busy,setBusy]=React.useState(false),[message,setMessage]=React.useState(''),[error,setError]=React.useState(''),[loaded,setLoaded]=React.useState(false);
 const [checking,setChecking]=React.useState(true);
 const [recoveryError,setRecoveryError]=React.useState('');
 const [pending,setPending]=React.useState(null),[receipt,setReceipt]=React.useState(null);
 const lock=React.useRef(false),alive=React.useRef(false),sequence=React.useRef(0);
 const key=proposalIntentKey(api,ticker,record.id);
 const refresh=async()=>{
  if(lock.current)return;
  const seq=++sequence.current;setChecking(true);
  const current=()=>alive.current&&seq===sequence.current;
  // Saved server links must remain accessible even when local recovery fails.
  try{
   const r=await fetch(`${api}/api/research/amendments/${encodeURIComponent(ticker)}?workId=${encodeURIComponent(record.id)}`,{signal:AbortSignal.timeout(20000)});
   if(!r.ok)throw Error('Proposal status unavailable. Previously loaded links remain available.');
   const d=linkedProposals(await r.json(),ticker,record.id);
   if(current()){setData(d);setError('');setLoaded(true);}
  }catch(e){if(current())setError(e.message);}
  if(!current())return;
  try{
   const saved=readProposalIntent(sessionStorage,key,record.id);
   if(current())setPending(saved);
   if(saved){
    const r=await fetch(`${api}/api/research/amendments/${encodeURIComponent(ticker)}?requestId=${encodeURIComponent(saved.requestId)}`,{signal:AbortSignal.timeout(20000)});
    if(!r.ok)throw Error('Exact proposal status unavailable. No submission was retried.');
    const found=proposalReceipt(await r.json(),ticker,saved);
    if(!current())return;
    if(found){
     setReceipt(found);
     clearProposalIntent(sessionStorage,key,record.id,saved);
     setPending(null);setMessage('Saved proposal found. No additional comparison was started.');
    }
   }
   if(current())setRecoveryError('');
  }catch(e){if(current())setRecoveryError('Local submission recovery is unresolved. '+(e.message||'Browser storage is unavailable.')+' Preparing research remains paused; saved server links are still available.');}finally{if(current())setChecking(false);}
 };
 React.useEffect(()=>{alive.current=true;refresh();const timer=setInterval(()=>{if(!document.hidden)refresh();},15000);return()=>{alive.current=false;sequence.current++;clearInterval(timer);};},[api,ticker,record.id]);
 const submit=async()=>{
  if(lock.current)return;
  lock.current=true;sequence.current++;setBusy(true);setMessage('');
  try{
   const payload=reserveProposalIntent(sessionStorage,key,record.id,{requestId:crypto.randomUUID(),workEvidence:{id:record.id,revision:record.revision},instructions:''});
   setPending(payload);
   const r=await fetch(`${api}/api/research/amendments/${encodeURIComponent(ticker)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:AbortSignal.timeout(20000)});
   const d=await r.json();
   if(!r.ok)throw Error(d.error||'Proposal submission not confirmed.');
   if(d.jobId!==payload.requestId)throw Error('Submission receipt did not match the saved request.');
   if(alive.current)setMessage('Submission received. Checking the exact saved proposal; no thesis edits were applied.');
  }catch(e){if(alive.current)setMessage(`${e.message} Check proposal status before retrying; any retry keeps the original saved request.`);}
  finally{lock.current=false;if(alive.current){await refresh();setBusy(false);}}
 };
 const linked=data.jobs,active=data.activeProposal;
 return <details className="workspace-panel"><summary>Recorded answer → thesis proposal</summary>
  <p>Compare this saved follow-up (revision {record.revision}) with the current thesis. Charlie will retain the meeting, question, recorded answer and your separate interpretation, then prepare proposed edits for review.</p>
  <p className="desk-explainer">Uses model credits only when you submit. Recorded notes are not independently verified transcripts. Changed answers or case baselines require an updated follow-up. The saved thesis stays unchanged until you explicitly accept edits.</p>
  {error&&<p role="alert">{error}</p>}{recoveryError&&<p role="alert">{recoveryError}</p>}
  {pending&&<p role="status">Unresolved submission for saved follow-up revision {pending.workEvidence.revision}{pending.workEvidence.revision!==record.revision?' (different from the current revision)':''}. Reloading only checks its status. An explicit retry reuses this exact request; it does not use newer answers.</p>}
  {receipt&&<p><a href={proposalLink(ticker,receipt.id)}>Review recovered proposal</a> · {receipt.status.replaceAll('_',' ')}</p>}
  {!loaded&&!error&&<p role="status">Loading linked proposals…</p>}{loaded&&!error&&!linked.length&&<p>No saved thesis proposal is linked to this follow-up. Other company proposals are separate.</p>}
  {linked.map(j=><p key={j.id}>Follow-up r{j.work_evidence.revision}{j.work_evidence.revision!==record.revision?' · Different saved revision':''} · {j.status.replaceAll('_',' ')}{j.error?` · ${j.error}`:''} · <a href={proposalLink(ticker,j.id)}>Review this saved proposal</a></p>)}
  {data.truncated&&<p>Showing the 30 most recent proposals for this follow-up. Older proposal links remain valid; this is not a complete history.</p>}
  {active&&<p>A thesis proposal already exists for this company. <a href={proposalLink(ticker,active.id)}>Review the active company proposal</a> before starting another. It may relate to different evidence.</p>}
  <button disabled={disabled||busy||checking||!!error||!!recoveryError||active||!loaded} onClick={submit}>{busy?'Submitting…':pending?'Retry original saved request':'Prepare thesis proposal from saved answer'}</button>{' '}
  <button disabled={busy} onClick={refresh}>Check proposal status</button>{' '}<a href={`#view=desk&section=evidence&ticker=${encodeURIComponent(ticker)}`}>All company proposals →</a>
  {message&&<p role="status">{message}</p>}
 </details>;
}
