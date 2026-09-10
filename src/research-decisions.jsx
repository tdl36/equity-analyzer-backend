import * as React from 'react';
const blank=()=>({decision:'',rationale:'',revisitWhen:'',decisionDate:new Date().toLocaleDateString('en-CA'),supersedes:''});
export function ResearchDecisions({api,ticker}) {
 const [draft,setDraft]=React.useState(blank),[rows,setRows]=React.useState([]),[revision,setRevision]=React.useState(null),[more,setMore]=React.useState(false),[busy,setBusy]=React.useState(false),[message,setMessage]=React.useState('');
 const [filters,setFilters]=React.useState({q:'',from:'',to:''}),[loading,setLoading]=React.useState(false),[inspected,setInspected]=React.useState(null),[chosen,setChosen]=React.useState(null);
 const applied=React.useRef({}),cursor=React.useRef(null),sequence=React.useRef(0);
 const alive=React.useRef(true),lock=React.useRef(false),pending=React.useRef(null);
 const json=async(options,query='')=>{const r=await fetch(`${api}/api/research/decisions/${encodeURIComponent(ticker)}${query}`,{...options,signal:AbortSignal.timeout(20000)});const d=await r.json();if(!r.ok){const e=Error(d.error||'Decision log unavailable');e.status=r.status;throw e;}return d;};
 const load=async(append=false,query=applied.current)=>{const seq=++sequence.current;setLoading(true);const params=new URLSearchParams(query);if(append&&cursor.current)params.set('before',cursor.current);
  try{const d=await json(undefined,'?'+params);if(alive.current&&seq===sequence.current){setRows(previous=>append?[...previous,...d.decisions.filter(r=>!previous.some(p=>p.id===r.id))]:d.decisions);setRevision(d.revision);setMore(d.hasMore);cursor.current=d.nextBefore;applied.current=query;}}
  catch(e){if(alive.current&&seq===sequence.current)setMessage(e.message);}
  finally{if(alive.current&&seq===sequence.current)setLoading(false);}
 };
 const inspect=async(id)=>{try{const d=await json(undefined,'?id='+encodeURIComponent(id));if(alive.current){setInspected(d.decisions[0]||null);if(!d.decisions.length)setMessage('Earlier decision not found.');}}catch(e){if(alive.current)setMessage(e.message);}};
 React.useEffect(()=>{alive.current=true;load();return()=>{alive.current=false;};},[]);
 const change=(key,value)=>{setDraft(d=>({...d,[key]:value}));pending.current=null;};
 const save=async()=>{if(lock.current||loading||revision===null)return;lock.current=true;setBusy(true);setMessage('Saving decision…');const payload=pending.current||{...draft,revision,requestId:crypto.randomUUID()};pending.current=payload;
  try{await json({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(alive.current){pending.current=null;setDraft(blank());setChosen(null);setMessage('Decision recorded. Existing research and portfolio positions were not changed.');await load();}}
  catch(e){if(e.status===409)pending.current=null;if(alive.current)setMessage(e.message+' Reload the log if needed; your draft is retained.');}
  finally{lock.current=false;if(alive.current)setBusy(false);}
 };
 return <details><summary>Decision log · {rows.length} loaded records</summary>
  <p className="desk-explainer">Record your reasoning and what would make you revisit it. Entries are retained; superseding a record preserves its history. These are research decisions, not trade instructions. Revisit conditions are not yet automatically monitored.</p>
  <fieldset disabled={busy||loading}><label>Decision date<input type="date" value={draft.decisionDate} onChange={e=>change('decisionDate',e.target.value)}/></label>
   {[['decision','My decision',2000],['rationale','Why I decided this',12000],['revisitWhen','Revisit when…',6000]].map(([key,label,limit])=><label key={key}>{label}<textarea rows={key==='decision'?2:3} maxLength={limit} value={draft[key]} onChange={e=>change(key,e.target.value)}/></label>)}
   <label>Does this replace an earlier decision?<select value={draft.supersedes} onChange={e=>{change('supersedes',e.target.value);setChosen(rows.find(r=>r.id===e.target.value)||null);}}><option value="">Independent new record</option>{[...rows,...(chosen&&!rows.some(r=>r.id===chosen.id)?[chosen]:[])].filter(r=>!r.superseded).map(r=><option key={r.id} value={r.id}>#{r.revision} · {r.body.decisionDate} · {r.body.decision.slice(0,90)}</option>)}</select></label>
   {chosen&&<p>Replacing #{chosen.revision}: {chosen.body.decision}</p>}
   <button className="workspace-primary" disabled={revision===null||!draft.decision.trim()||!draft.rationale.trim()||!draft.revisitWhen.trim()} onClick={save}>{busy?'Saving…':'Record decision'}</button>
   <button onClick={()=>load()}>Reload log · keep draft</button>
  </fieldset>
  <fieldset disabled={busy||loading}><legend>Find earlier decisions</legend>
   <label>Search decisions and reasoning<input maxLength={200} value={filters.q} onChange={e=>setFilters({...filters,q:e.target.value})} placeholder="Cash conversion, pricing, management…"/></label>
   <label>Decision date from<input type="date" value={filters.from} onChange={e=>setFilters({...filters,from:e.target.value})}/></label>
   <label>Decision date through<input type="date" value={filters.to} onChange={e=>setFilters({...filters,to:e.target.value})}/></label>
   <button onClick={()=>load(false,filters)}>Search history</button><button onClick={()=>{setFilters({q:'',from:'',to:''});load(false,{});}}>Clear filters</button>
  </fieldset>
  {loading&&<p role="status">Loading decision history…</p>}
  {inspected&&<aside style={{border:'1px solid var(--border, #bbb)',padding:12}}><strong>Earlier record #{inspected.revision} · {inspected.body.decisionDate} · {inspected.superseded?'Superseded':'Recorded decision'}</strong><h4>{inspected.body.decision}</h4><p style={{whiteSpace:'pre-wrap'}}>{inspected.body.rationale}</p><p>Revisit when: {inspected.body.revisitWhen}</p>{inspected.body.supersedes&&<button onClick={()=>inspect(inspected.body.supersedes)}>Read preceding decision</button>}<button onClick={()=>setInspected(null)}>Close earlier record</button></aside>}
  {message&&<p role="status">{message}</p>}
  {rows.map(r=><article key={r.id} style={{borderTop:'1px solid var(--border, #bbb)',padding:'12px 0'}}><strong>#{r.revision} · {r.body.decisionDate} · {r.superseded?'Superseded': 'Recorded decision'}</strong><h4>{r.body.decision}</h4><p style={{whiteSpace:'pre-wrap'}}>{r.body.rationale}</p><p style={{whiteSpace:'pre-wrap'}}><strong>Revisit when: </strong>{r.body.revisitWhen}</p>{r.body.supersedes&&<button onClick={()=>inspect(r.body.supersedes)}>Read the decision this superseded</button>}<p><small>Recorded {new Date(r.created_at).toLocaleString()}</small></p></article>)}
  {!rows.length&&revision!==null&&<p>No decisions match this view for {ticker}. Clear filters to see recent records.</p>}
  {more&&<button disabled={loading||busy} onClick={()=>load(true)}>Load older matching decisions</button>}<p className="desk-explainer">History is ordered by when entries were recorded. Date filters use the decision date. Shared AI context still includes only the latest 20 records; searching here does not change an agent’s context.</p>
 </details>;
}
