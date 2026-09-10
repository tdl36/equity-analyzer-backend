import * as React from 'react';
const blank=()=>({decision:'',rationale:'',revisitWhen:'',decisionDate:new Date().toLocaleDateString('en-CA'),supersedes:''});
export function ResearchDecisions({api,ticker}) {
 const [draft,setDraft]=React.useState(blank),[rows,setRows]=React.useState([]),[revision,setRevision]=React.useState(null),[more,setMore]=React.useState(false),[busy,setBusy]=React.useState(false),[message,setMessage]=React.useState('');
 const alive=React.useRef(true),lock=React.useRef(false),pending=React.useRef(null);
 const json=async(options)=>{const r=await fetch(`${api}/api/research/decisions/${encodeURIComponent(ticker)}`,{...options,signal:AbortSignal.timeout(20000)});const d=await r.json();if(!r.ok){const e=Error(d.error||'Decision log unavailable');e.status=r.status;throw e;}return d;};
 const load=async()=>{try{const d=await json();if(alive.current){setRows(d.decisions);setRevision(d.revision);setMore(d.hasMore);}}catch(e){if(alive.current)setMessage(e.message);}};
 React.useEffect(()=>{alive.current=true;load();return()=>{alive.current=false;};},[]);
 const change=(key,value)=>{setDraft(d=>({...d,[key]:value}));pending.current=null;};
 const save=async()=>{if(lock.current||revision===null)return;lock.current=true;setBusy(true);setMessage('Saving decision…');const payload=pending.current||{...draft,revision,requestId:crypto.randomUUID()};pending.current=payload;
  try{await json({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(alive.current){pending.current=null;setDraft(blank());setMessage('Decision recorded. Existing research and portfolio positions were not changed.');await load();}}
  catch(e){if(e.status===409)pending.current=null;if(alive.current)setMessage(e.message+' Reload the log if needed; your draft is retained.');}
  finally{lock.current=false;if(alive.current)setBusy(false);}
 };
 return <details><summary>Decision log · {rows.length} recent records</summary>
  <p className="desk-explainer">Record your reasoning and what would make you revisit it. Entries are retained; superseding a record preserves its history. These are research decisions, not trade instructions. Revisit conditions are not yet automatically monitored.</p>
  <fieldset disabled={busy}><label>Decision date<input type="date" value={draft.decisionDate} onChange={e=>change('decisionDate',e.target.value)}/></label>
   {[['decision','My decision',2000],['rationale','Why I decided this',12000],['revisitWhen','Revisit when…',6000]].map(([key,label,limit])=><label key={key}>{label}<textarea rows={key==='decision'?2:3} maxLength={limit} value={draft[key]} onChange={e=>change(key,e.target.value)}/></label>)}
   <label>Does this replace an earlier decision?<select value={draft.supersedes} onChange={e=>change('supersedes',e.target.value)}><option value="">Independent new record</option>{rows.filter(r=>!r.superseded).map(r=><option key={r.id} value={r.id}>#{r.revision} · {r.body.decisionDate} · {r.body.decision.slice(0,90)}</option>)}</select></label>
   <button className="workspace-primary" disabled={revision===null||!draft.decision.trim()||!draft.rationale.trim()||!draft.revisitWhen.trim()} onClick={save}>{busy?'Saving…':'Record decision'}</button>
   <button onClick={load}>Reload log · keep draft</button>
  </fieldset>
  {message&&<p role="status">{message}</p>}
  {rows.map(r=><article key={r.id} style={{borderTop:'1px solid var(--border, #bbb)',padding:'12px 0'}}><strong>#{r.revision} · {r.body.decisionDate} · {r.superseded?'Superseded': 'Recorded decision'}</strong><h4>{r.body.decision}</h4><p style={{whiteSpace:'pre-wrap'}}>{r.body.rationale}</p><p style={{whiteSpace:'pre-wrap'}}><strong>Revisit when: </strong>{r.body.revisitWhen}</p>{r.body.supersedes&&<small>Replaces an earlier recorded decision. Original retained.</small>}<p><small>Recorded {new Date(r.created_at).toLocaleString()}</small></p></article>)}
  {!rows.length&&revision!==null&&<p>No decisions recorded for {ticker} yet.</p>}
  {more&&<p>Showing the latest 50 records. Older records remain stored; shared memory includes the latest 20.</p>}
 </details>;
}
