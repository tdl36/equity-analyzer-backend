import * as React from 'react';
import {synthesisRows} from './collection-synthesis-model.mjs';
export function CollectionSynthesisStatus({api}) {
 const [data,setData]=React.useState({}),[checked,setChecked]=React.useState(null);
 React.useEffect(()=>{
  let live=true,pending=false;const controller=new AbortController();setData({});setChecked(null);
  const refresh=async()=>{
   if(pending)return;pending=true;
   const results=await Promise.allSettled(['/api/catalysts/auto-mode','/api/research/automation','/api/analysts'].map(async path=>{
    const r=await fetch(`${api}${path}`,{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(20000)])});
    if(!r.ok)throw new Error('Status unavailable');return r.json();
   }));
   if(live){setData(Object.fromEntries(['catalysts','intake','analysts'].map((key,i)=>[key,results[i].status==='fulfilled'?results[i].value:null])));setChecked(new Date());}pending=false;
  };
  refresh();const timer=setInterval(()=>{if(!document.hidden)refresh();},30000);
  return()=>{live=false;controller.abort();clearInterval(timer);};
 },[api]);
 return <section className="workspace-panel" aria-label="Automatic synthesis settings"><h3>Automatic synthesis</h3>
  <p>Collection and importing can continue with these switches off. During backfill, keep all three off to avoid automatic generation from new documents.</p>
  {synthesisRows(data).map(row=><p key={row.label}><strong>{row.label}: {row.status}</strong>{' · '}<a href={row.href}>{row.control}</a></p>)}
  <p className="desk-explainer">{checked?`Last checked ${checked.toLocaleTimeString()}.`:'Checking saved settings…'} These are separate controls. Pausing them does not cancel work already running or change outputs explicitly requested in an assignment. Open the linked settings to change them.</p>
 </section>;
}
