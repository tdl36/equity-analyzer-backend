import * as React from 'react';
import {latestWorkForComparison,workDifferences} from './work-comparison.mjs';
// Read-only: never replace the editor, advance its baseline or submit work.
export function WorkComparison({api,ticker,record,draft,baseline,onOpen,replaceDisabled=false}){
 const [state,setState]=React.useState(null),alive=React.useRef(true),sequence=React.useRef(0);
 React.useEffect(()=>{alive.current=true;return()=>{alive.current=false;++sequence.current;};},[]);
 const load=async()=>{
  const seq=++sequence.current;setState({loading:true});
  try{
   const response=await fetch(api+'/api/research/workbench/'+encodeURIComponent(ticker),{cache:'no-store',signal:AbortSignal.timeout(20000)});
   if(!response.ok)throw Error('Saved work is unavailable. Retry the comparison; your draft is unchanged.');
   const saved=latestWorkForComparison(await response.json(),record,ticker,draft.kind);
   if(alive.current&&seq===sequence.current)setState({saved,checked:new Date().toLocaleTimeString()});
  }catch(e){if(alive.current&&seq===sequence.current)setState({error:e.message});}
 };
 const rows=state?.saved?workDifferences(draft,state.saved.draft):[];
 return <section className="workspace-panel" aria-label="Saved work comparison">
 <button disabled={state?.loading} onClick={load}>{state?'Refresh saved work comparison':'Compare draft with latest saved work'}</button>
 {state&&<button onClick={()=>{++sequence.current;setState(null);}}>Close comparison</button>}
 {state?.loading&&<p role="status">Reading saved work… Your draft stays unchanged.</p>}
 {state?.error&&<p role="alert">{state.error}</p>}
 {state?.saved&&<><h3>Draft versus saved work</h3><p>Your draft is based on work revision {record.revision}, case revision {baseline}. Latest saved work at {state.checked}: revision {state.saved.revision}, case revision {state.saved.caseRevision}.</p>
 {state.saved.revision>record.revision&&<p role="alert">A newer work revision is saved. Your editing baseline has not changed. Review the differences before deciding what to keep.</p>}
 <p className="desk-explainer">Read-only comparison of editable fields. Derived calculations and saved evidence snapshots remain in the saved record and version history. This read does not establish that sources or the investment case are current. Refresh to check for subsequent saves.</p>
 {!rows.length?<p>No editable field differences. Revision and case baselines above still apply.</p>:<><p>{rows.length} differing fields. Unchanged fields are omitted.</p>{rows.map(r=><div key={r.key} className="workspace-panel" style={{overflowWrap:'anywhere'}}><h4>{r.label}</h4><div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit, minmax(min(100%, 260px), 1fr))',gap:'1rem'}}><div style={{minWidth:0}}><strong>Your draft</strong><p style={{whiteSpace:'pre-wrap'}}>{r.local}</p></div><div style={{minWidth:0}}><strong>Saved revision {state.saved.revision}</strong><p style={{whiteSpace:'pre-wrap'}}>{r.saved}</p></div></div></div>)}</>}
 <p>Keep editing your draft, or open the compared saved version. Replacing unfinished edits requires confirmation. Nothing is merged or saved by this comparison.</p><button disabled={replaceDisabled} onClick={()=>onOpen({id:state.saved.id,revision:state.saved.revision},state.saved.draft)}>Open compared saved revision {state.saved.revision}</button></>}
 </section>;
}
