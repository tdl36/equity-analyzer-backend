import {linkedProposals,proposalLink} from './meeting-answer-proposal-model.mjs';
import * as React from 'react';
export function MeetingAnswerProposal(props) {return <ProposalHandoff key={props.api+'|'+props.ticker+'|'+props.record.id} {...props}/>;}
function ProposalHandoff({api,ticker,record,disabled=false}) {
 const [data,setData]=React.useState({jobs:[],activeProposal:null,truncated:false}),[busy,setBusy]=React.useState(false),[message,setMessage]=React.useState(''),[error,setError]=React.useState(''),[loaded,setLoaded]=React.useState(false);
 const lock=React.useRef(false),pending=React.useRef(null),alive=React.useRef(false),sequence=React.useRef(0);
 const refresh=async()=>{const seq=++sequence.current;try{const r=await fetch(`${api}/api/research/amendments/${encodeURIComponent(ticker)}?workId=${encodeURIComponent(record.id)}`,{signal:AbortSignal.timeout(20000)});if(!r.ok)throw Error('Proposal status unavailable. Previously loaded links remain available.');const d=linkedProposals(await r.json(),ticker,record.id);if(alive.current&&seq===sequence.current){setData(d);setError('');setLoaded(true);}}catch(e){if(alive.current&&seq===sequence.current)setError(e.message);}};
 React.useEffect(()=>{alive.current=true;refresh();const timer=setInterval(()=>{if(!document.hidden)refresh();},15000);return()=>{alive.current=false;sequence.current++;clearInterval(timer);};},[api,ticker,record.id]);
 const submit=async()=>{if(lock.current)return;lock.current=true;setBusy(true);setMessage('');
  const payload=pending.current||{requestId:crypto.randomUUID(),workEvidence:{id:record.id,revision:record.revision},instructions:''};pending.current=payload;
  try{const r=await fetch(`${api}/api/research/amendments/${ticker}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:AbortSignal.timeout(20000)});const d=await r.json();if(!r.ok){if(r.status<500)pending.current=null;throw Error(d.error||'Proposal submission not confirmed.');}pending.current=null;if(alive.current)setMessage('Thesis comparison queued. Review the proposed wording in Evidence & changes; no thesis edits were applied.');}
  catch(e){if(alive.current)setMessage(`${e.message}${pending.current?' Use Check status first; submitting again reuses the same request identity.':''}`);}
  finally{if(alive.current){await refresh();setBusy(false);}lock.current=false;}
 };
 const linked=data.jobs,active=data.activeProposal;
 return <details className="workspace-panel"><summary>Recorded answer → thesis proposal</summary>
  <p>Compare this saved follow-up (revision {record.revision}) with the current thesis. Charlie will retain the meeting, question, recorded answer and your separate interpretation, then prepare proposed edits for review.</p>
  <p className="desk-explainer">Uses model credits only when you submit. Recorded notes are not independently verified transcripts. Changed answers or case baselines require an updated follow-up. The saved thesis stays unchanged until you explicitly accept edits.</p>
  {error&&<p role="alert">{error}</p>}
  {!loaded&&!error&&<p role="status">Loading linked proposals…</p>}{loaded&&!error&&!linked.length&&<p>No saved thesis proposal is linked to this follow-up. Other company proposals are separate.</p>}
  {linked.map(j=><p key={j.id}>Follow-up r{j.work_evidence.revision}{j.work_evidence.revision!==record.revision?' · Different saved revision':''} · {j.status.replaceAll('_',' ')}{j.error?` · ${j.error}`:''} · <a href={proposalLink(ticker,j.id)}>Review this saved proposal</a></p>)}
  {data.truncated&&<p>Showing the 30 most recent proposals for this follow-up. Older proposal links remain valid; this is not a complete history.</p>}
  {active&&<p>A thesis proposal already exists for this company. <a href={proposalLink(ticker,active.id)}>Review the active company proposal</a> before starting another. It may relate to different evidence.</p>}
  <button disabled={disabled||busy||!!error||active||!loaded} onClick={submit}>{busy?'Submitting…':pending.current?'Reconcile same proposal request':'Prepare thesis proposal from saved answer'}</button>{' '}
  <button disabled={busy} onClick={refresh}>Check proposal status</button>{' '}<a href={`#view=desk&section=evidence&ticker=${encodeURIComponent(ticker)}`}>All company proposals →</a>
  {message&&<p role="status">{message}</p>}
 </details>;
}
