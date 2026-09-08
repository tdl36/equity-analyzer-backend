import {AssignmentWorkspace} from './assignment-workspace';
import * as React from 'react';
const today=()=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York'}).format(new Date());
const focusOptions=[['thesis','Investment thesis'],['earnings','Earnings & guidance'],['competition','Competition & demand'],['capital','Capital allocation'],['followups','Previous meeting follow-ups']];
const pendingKey='charlie-meeting-request-v1';
function stored(){try{return JSON.parse(sessionStorage.getItem(pendingKey)||'null');}catch{return null;}}
export function MeetingCommand({api,onNavigate,onSection,sourcePolicy}){
  const [tickers,setTickers]=React.useState([]),[selected,setSelected]=React.useState([]),[query,setQuery]=React.useState('');
  const [meetingDate,setMeetingDate]=React.useState(today),[days,setDays]=React.useState(90),[focuses,setFocuses]=React.useState(['thesis','earnings','followups']),[note,setNote]=React.useState('');
  const [jobs,setJobs]=React.useState([]),[busy,setBusy]=React.useState(false),[error,setError]=React.useState(''),[message,setMessage]=React.useState(''),[uncertain,setUncertain]=React.useState(()=>!!stored());
  const pending=React.useRef(stored()),lock=React.useRef(false),alive=React.useRef(true);
  const json=async(path,options={})=>{const r=await fetch(api+path,{...options,signal:AbortSignal.timeout(20000)});let d;try{d=await r.json();}catch{throw Error('The server response could not be confirmed.');}if(!r.ok){const e=Error(d.error||`Request failed (${r.status})`);e.status=r.status;throw e;}return d;};
  const clear=()=>{pending.current=null;try{sessionStorage.removeItem(pendingKey);}catch{}setUncertain(false);};
  const refresh=async()=>{const d=await json('/api/research/meeting-commands');if(!alive.current)return;setTickers(d.tickers);setJobs(d.jobs);setError('');if(pending.current&&d.jobs.some(j=>j.batch_id===pending.current.requestId)){clear();setMessage('Your meeting assignment is recorded. Follow each company below.');}};
  React.useEffect(()=>{alive.current=true;refresh().catch(e=>setError(e.message));const timer=setInterval(()=>{if(!document.hidden)refresh().catch(e=>{if(alive.current)setError(e.message);});},10000);return()=>{alive.current=false;clearInterval(timer);};},[api]);
  const toggle=(v,list,set)=>set(list.includes(v)?list.filter(x=>x!==v):[...list,v]);
  const submit=async()=>{if(lock.current)return;lock.current=true;setBusy(true);setError('');
    const body=pending.current||{requestId:crypto.randomUUID(),tickers:selected,meetingDate,date:today(),days,focuses,note,sourcePolicy};
    pending.current=body;try{sessionStorage.setItem(pendingKey,JSON.stringify(body));}catch{}
    try{const d=await json('/api/research/meeting-commands',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});clear();setMessage(`Queued ${d.commands.length} meeting pack${d.commands.length===1?'':'s'}. You can leave this page; progress is saved.`);await refresh();}
    catch(e){setError(e.message);if(e.status&&e.status<500)clear();else setUncertain(true);}finally{lock.current=false;setBusy(false);}
  };
  const retry=async id=>{if(lock.current)return;lock.current=true;setBusy(true);try{await json(`/api/research/meeting-commands/${id}/retry`,{method:'POST'});await refresh();setMessage('Meeting pack queued to resume from its saved checkpoint.');}catch(e){setError(e.message);}finally{lock.current=false;setBusy(false);}};
  const issueFor=j=>j.prep_error||j.error||(j.meeting_issue==='The latest linked analyst activity has no completed recap yet'?'':j.meeting_issue);
  return <section className="workspace-panel meeting-command">
    <div className="meeting-command-heading"><div><p className="workspace-eyebrow">MEETING & CONFERENCE PREP</p><h2>Walk in with better questions.</h2><p className="desk-explainer">Choose your companies. Charlie collects recent research, prepares a brief and saves questions in Meeting Prep. No prompt required.</p></div><span className="meeting-command-count">{selected.length}<small>of 10 companies</small></span></div>
    <fieldset disabled={busy||uncertain} className="meeting-command-form">
      <div className="meeting-command-step"><h3><span>1</span> Choose companies</h3><label>Find a covered company<input type="search" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search ticker, e.g. ABT"/></label>
        <div className="meeting-ticker-grid">{tickers.filter(t=>t.toLowerCase().includes(query.toLowerCase())).map(t=><label key={t} className={selected.includes(t)?'selected':''}><input type="checkbox" checked={selected.includes(t)} disabled={selected.length>=10&&!selected.includes(t)} onChange={()=>toggle(t,selected,setSelected)}/>{t}</label>)}</div>
        {!tickers.length&&<p>Coverage loads from your analyst team. Assign a covering analyst before requesting a pack.</p>}
        {selected.length>0&&<p className="meeting-selection">Selected: <strong>{selected.join(' · ')}</strong> <button type="button" onClick={()=>setSelected([])}>Clear selection</button></p>}
      </div>
      <div className="meeting-command-step"><h3><span>2</span> Set the brief</h3><div className="desk-filter"><label>Meeting date<input type="date" value={meetingDate} onChange={e=>setMeetingDate(e.target.value)}/></label><label>Source lookback<select value={days} onChange={e=>setDays(Number(e.target.value))}><option value={30}>Past 30 days</option><option value={60}>Past 60 days</option><option value={90}>Past 90 days · default</option></select></label></div>
        <p>What matters most? <small>Optional—start with the selected defaults.</small></p><div className="meeting-focuses">{focusOptions.map(([id,label])=><label key={id}><input type="checkbox" checked={focuses.includes(id)} onChange={()=>toggle(id,focuses,setFocuses)}/>{label}</label>)}</div>
        <label>Anything specific? <small>Optional</small><textarea maxLength={1000} value={note} onChange={e=>setNote(e.target.value)} placeholder="For example: focus on FreeStyle Libre adoption and margin durability."/></label>
      </div>
      <div className="meeting-command-preview"><p className="workspace-eyebrow">YOUR ASSIGNMENT</p><p>Prepare {selected.length?selected.join(', '):'your selected companies'} for {meetingDate||'your meeting date'}, using sources from the past {days} days. Seek earnings transcripts, presentations and sell-side reaction. Explain changes and prepare prioritized questions with source references and follow-ups.</p><p className="desk-explainer">Collection → verified originals → analyst brief → meeting questions. Uses model credits for each company. Your Mac, signed-in AlphaSense browser and server research key are required. Browser pickup is scheduled, not instant; missing presentations or other sources are disclosed.</p><button className="workspace-primary" type="button" disabled={!sourcePolicy||!selected.length||!meetingDate} onClick={submit}>{busy?'Queuing…':selected.length>1?`Prepare ${selected.length} company packs →`:'Prepare meeting pack →'}</button></div>
    </fieldset>
    {uncertain&&<div role="status"><p>Submission is not yet confirmed. Retry checks the same request; it does not create a new batch.</p><button disabled={busy} onClick={submit}>Check / retry same assignment</button></div>}
    {error&&<p role="alert" className="workspace-error">{error}</p>}{message&&<p role="status">{message}</p>}
    <div className="workspace-section-heading"><h3>Your meeting packs</h3><button onClick={()=>onNavigate('meetingprep')}>Open Meeting Prep →</button></div>
    {!jobs.length&&<p className="workspace-empty">Your assignments will appear here, with separate progress for every company.</p>}
    <div className="meeting-pack-list">{jobs.map(j=><article key={j.id}><AssignmentWorkspace api={api} id={j.id} onNavigate={onNavigate} onSection={onSection}/><div><strong>{j.ticker}</strong><span>{j.options?.meetingPrep?.meetingDate}</span></div><p>{j.prep_status==='done'?'Pack ready':j.prep_status==='running'?`Preparing questions · ${j.prep_step||'starting'}${j.completed?` (${j.completed}/${j.total})`:''}`:j.prep_status==='queued'?'Sources verified · waiting for preparation':j.prep_status==='failed'?'Preparation needs attention':j.status==='failed'?'Collection request failed':j.status==='queued'?'Waiting for Mac acknowledgment':'Collecting sources / preparing analyst brief'}</p>{issueFor(j)&&<p className="workspace-error">{issueFor(j)}</p>}{j.meeting_id&&<button onClick={()=>{window.dispatchEvent(new CustomEvent('charlie-open-meeting',{detail:{id:Number(j.meeting_id)}}));onNavigate('meetingprep');}}>Open {j.ticker} meeting →</button>}{j.prep_status==='failed'&&<button disabled={busy} onClick={()=>retry(j.job_id)}>Retry meeting pack</button>}<button onClick={()=>onSection('collection')}>Collection details →</button></article>)}</div>
  </section>;
}
