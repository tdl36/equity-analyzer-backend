import * as React from 'react';
import {historyPage,appendHistory} from './assignment-history-model.mjs';
import {activityAge} from './activity-model.mjs';
import {routeHash} from './workspace-model.mjs';

export function AssignmentHistory({api}) {
 const [open,setOpen]=React.useState(false),[query,setQuery]=React.useState(''),[filter,setFilter]=React.useState('');
 const [state,setState]=React.useState({rows:[],cursor:null,loaded:false,busy:false});
 const generation=React.useRef(0),active=React.useRef(null),locked=React.useRef(false);
 async function load(more=false) {
  if(locked.current)return;
  locked.current=true;const token=++generation.current,c=new AbortController();active.current=c;
  const cursor=more?state.cursor:null;
  setState(s=>({...s,busy:true,error:''}));
  try {
   const params=new URLSearchParams();if(filter)params.set('ticker',filter);if(cursor)params.set('cursor',cursor);
   const response=await fetch(`${api}/api/research/assignment-history?${params}`,{signal:AbortSignal.any([c.signal,AbortSignal.timeout(30000)])});
   const data=await response.json();if(!response.ok)throw Error(data.error||'Saved assignments are unavailable.');
   const page=historyPage(data,filter);
   if(token===generation.current)setState(s=>({rows:more?appendHistory(s.rows,page.rows):page.rows,cursor:page.cursor,loaded:true,busy:false}));
  }catch(error){if(token===generation.current)setState(s=>({...s,busy:false,error:'Could not load saved assignments. Your loaded rows remain available. Retry this read.'}));}
  finally{if(token===generation.current)locked.current=false;}
 }
 React.useEffect(()=>{
  generation.current++;active.current?.abort();locked.current=false;
  setState({rows:[],cursor:null,loaded:false,busy:false});
  if(open)load();
  return()=>{generation.current++;active.current?.abort();locked.current=false;};
 },[api,open,filter]);
 const statuses={complete:'Finished · review saved outputs',completed:'Finished · review saved outputs',attention:'Needs attention',running:'In progress',queued:'Waiting',cancelled:'Stopped',stopped:'Stopped'};
 return <section className="activity-archive" aria-label="Saved assignment archive">
  <button onClick={()=>setOpen(v=>!v)} aria-expanded={open}>{open?'Close assignment archive':'Find older assignments'}</button>
  {open&&<>
   <h3>Saved assignments</h3><p>Browse across dates and companies. Finished work still requires your review.</p>
   <form className="activity-toolbar" onSubmit={e=>{e.preventDefault();const value=query.trim().toUpperCase();if(value===filter)load();else setFilter(value);}}>
    <label>Company ticker <input value={query} maxLength={30} onChange={e=>setQuery(e.target.value)} placeholder="All companies" aria-label="Archive company ticker" pattern="[A-Za-z0-9.\-]*"/></label>
    <button type="submit" disabled={state.busy}>Find assignments</button>
   </form>
   {state.error&&<p role="alert">{state.error}</p>}
   {state.busy&&<p role="status">Loading saved assignments…</p>}
   {state.loaded&&!state.rows.length&&<p>No saved assignments match this company filter.</p>}
   {state.rows.map(row=><article className="activity-row" key={row.id}>
    <div className="activity-company"><strong>{row.ticker}</strong><small>{statuses[row.status]||'Inspect saved status'}</small></div>
    <div className="activity-detail"><small>Created {activityAge(row.created_at)} · Updated {activityAge(row.updated_at)}</small></div>
    <a className="activity-open" href={routeHash('stockanalysis',row.ticker,'stock-analysis','assignment:'+row.id)}>Open saved assignment ↗<span className="sr-only"> {row.ticker}</span></a>
   </article>)}
   {state.cursor&&<button disabled={state.busy} onClick={()=>load(true)}>{state.error?'Retry older assignments':'Load older assignments'}</button>}
   {state.error&&!state.cursor&&<button disabled={state.busy} onClick={()=>load()}>Retry archive</button>}
  </>}
 </section>;
}
