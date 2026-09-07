import * as React from 'react';
const {useState}=React;
const date=v=>v?String(v).slice(0,10):'Date unavailable';
// Render stored structured research as text, never as executable HTML.
function ResearchText({value,depth=0}) {
  if(value==null||value==='')return null;
  if(typeof value!=='object')return <p style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{String(value)}</p>;
  if(depth>5)return <p>Open the company research to inspect additional detail.</p>;
  if(Array.isArray(value))return <div>{value.map((v,i)=><ResearchText key={i} value={v} depth={depth+1}/>)}</div>;
  return <div>{Object.entries(value).filter(([key,v])=>!key.startsWith('_')&&v!=null&&v!=='').map(([key,v])=><section className="evidence-saved-field" key={key}><h4>{key.replaceAll('_',' ').replace(/([a-z])([A-Z])/g,'$1 $2')}</h4><ResearchText value={v} depth={depth+1}/></section>)}</div>;
}
export function SavedResearchContext({data,ticker,onCompany}) {
  const thesis=data?.savedThesis,documents=data?.documents;
  const [all,setAll]=useState(false);
  const uploaded=documents?.uploaded||[],local=documents?.local||[];
  const items=[...uploaded.map(d=>({...d,location:'Saved in Charlie'})),...local.map(d=>({...d,location:`iCloud inventory · ${d.folder==='main'?'STOCKS':d.folder||'Folder unspecified'}`}))];
  return <div className="evidence-layout evidence-saved-context"><section className="workspace-panel evidence-main">
    <div><p className="workspace-eyebrow">SAVED COMPANY RESEARCH</p><h3>{thesis?`${ticker} · Saved investment thesis`:`${ticker} · Research sources`}</h3>{thesis&&<p className="desk-explainer">{thesis.company} · Updated {date(thesis.updatedAt)}</p>}</div>
    {!data.current&&<div className="workspace-notice"><strong>{thesis?'Your saved thesis is available.':'No saved thesis found.'}</strong><p>No investment-review comparison baseline exists yet. Available documents below are an inventory, not verified support for individual claims.</p></div>}
    {thesis?<><ResearchText value={thesis.thesis}/>{[['Signposts',thesis.signposts],['Threats',thesis.threats],['Conclusion',thesis.conclusion]].map(([label,value])=>value&&<details key={label}><summary>{label}</summary><ResearchText value={value}/></details>)}</>:<p className="desk-explainer">You can inspect available sources and open the company workspace without generating a report.</p>}
    <button className="workspace-primary" onClick={()=>onCompany(ticker,'portfolio')}>Open {ticker} research →</button>
  </section><aside className="workspace-panel evidence-sidebar"><div><p className="workspace-eyebrow">AVAILABLE SOURCE DOCUMENTS</p><h3>{uploaded.length} saved in Charlie · {local.length} in iCloud inventory</h3><p className="desk-explainer">Locations are listed separately; the same document may appear in both. Presence here does not mean a review has read or verified it.</p><p className="desk-explainer">{documents?.localUpdatedAt?`iCloud inventory reported ${String(documents.localUpdatedAt).replace('T',' ')}. This is the agent’s last report, not a live folder scan.`:'The local agent has not reported an iCloud inventory to this server yet. This does not mean your folders are empty.'}</p></div>
    {!items.length&&<p className="desk-explainer">No document entries are currently available in Charlie’s inventories.</p>}
    {(all?items:items.slice(0,20)).map((d,i)=><article className="evidence-document" key={`${d.location}-${d.filename}-${i}`}><strong>{d.filename||'Unnamed document'}</strong><small>{d.location}{d.addedAt?` · Added ${date(d.addedAt)}`:''}</small></article>)}
    {items.length>20&&<button onClick={()=>setAll(v=>!v)}>{all?'Show first 20':`Show all ${items.length} inventory entries`}</button>}
  </aside></div>;
}
