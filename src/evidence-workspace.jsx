import {ResearchAutomationControl} from './research-automation';
import {ThesisAmendments} from './thesis-amendments';
import {SavedResearchContext} from './saved-research-context';
import * as React from 'react';
const {useState,useEffect,useRef}=React;
const date=value=>value?String(value).slice(0,10):'Undated';
export function EvidenceWorkspace({api,analyses,onCompany}) {
  const [ticker,setTicker]=useState('DE'),[input,setInput]=useState('DE');
  const [data,setData]=useState(null),[error,setError]=useState(''),[loading,setLoading]=useState(true);
  const [selected,setSelected]=useState(null),[refresh,setRefresh]=useState(0);
  const inspector=useRef(null);
  const inspect=claim=>{setSelected(claim);requestAnimationFrame(()=>inspector.current?.scrollIntoView({block:'nearest',behavior:'smooth'}));};
  useEffect(()=>{
    const controller=new AbortController();let current=true;
    setLoading(true);setError('');setData(null);setSelected(null);
    const timer=setTimeout(()=>controller.abort(),20000);
    fetch(`${api}/api/research/evidence/${encodeURIComponent(ticker)}`,{signal:controller.signal})
      .then(async r=>{if(!r.ok)throw new Error(r.status===401?'Sign in to view saved research.':`Could not load research (${r.status}).`);return r.json();})
      .then(value=>{if(current)setData(value);}).catch(e=>{if(current)setError(e.name==='AbortError'?'The request timed out. Please retry.':e.message);})
      .finally(()=>{clearTimeout(timer);if(current)setLoading(false);});
    return()=>{current=false;clearTimeout(timer);controller.abort();};
  },[api,ticker,refresh]);
  const review=data?.current,prior=data?.prior,claims=review?.evidence?.claims||[];
  const sources=review?.evidence?.sources||[];
  const pick=path=>{const claim=claims.find(c=>c.path===path);inspect(claim||{statement:'No source passage was recorded for this item.',evidence:[]});};
  const showCompany=()=>onCompany(ticker,'portfolio');
  return <section className="evidence-workspace">
    <div className="evidence-intro"><div><p className="workspace-eyebrow">RESEARCH / EVIDENCE & CHANGES</p><h2>Read the change. Inspect the evidence.</h2><p className="desk-explainer">Read your saved thesis, inspect available documents, and compare investment reviews when a baseline exists.</p></div>
    <form className="evidence-search" onSubmit={e=>{e.preventDefault();const tk=input.trim().toUpperCase();if(/^[A-Z0-9][A-Z0-9.\-]{0,19}$/.test(tk)){setTicker(tk);setRefresh(x=>x+1);}else setError('Enter a valid ticker.');}}><label htmlFor="evidence-ticker">Company ticker</label><div><input id="evidence-ticker" list="evidence-companies" value={input} onChange={e=>setInput(e.target.value)} maxLength={20}/><button className="workspace-primary" type="submit">Open →</button></div><datalist id="evidence-companies">{[...new Set((analyses||[]).map(a=>a.ticker).filter(Boolean))].map(t=><option key={t} value={t}/>)}</datalist></form></div>
    <ResearchAutomationControl api={api}/>
    {!loading&&!error&&data&&<ThesisAmendments key={ticker} api={api} ticker={ticker} context={data} onApplied={()=>setRefresh(x=>x+1)}/>}
    {loading?<p role="status" className="workspace-empty">Loading saved research for {ticker}…</p>:error?<div className="workspace-error" role="alert">{error} <button onClick={()=>setRefresh(x=>x+1)}>Retry</button></div>:!review?<SavedResearchContext key={ticker} data={data} ticker={ticker} onCompany={onCompany}/>:<>
      <details className="workspace-panel evidence-context-toggle"><summary>Saved company thesis & available documents</summary><SavedResearchContext key={ticker} data={data} ticker={ticker} onCompany={onCompany}/></details>
      <div className="evidence-summary"><div><strong>{ticker}</strong><span>Review · {date(review.createdAt)}</span><span>{prior?`Compared with ${date(prior.createdAt)}`:'First saved review · no baseline'}</span></div><span className="evidence-badge">{review.quality.status==='checks_passed'?'Automated checks passed · analyst review required':'Needs review'}</span><button onClick={showCompany}>Open company ↗</button></div>
      <div className="evidence-layout"><main className="evidence-main">
        <section className="workspace-panel"><p className="workspace-eyebrow">01 / INVESTMENT JUDGMENT</p><h3>The current thesis</h3>{(review.state.thesis||[]).map((text,i)=><button className="evidence-claim" key={i} onClick={()=>pick(`thesis.${i}`)}><span>{String(i+1).padStart(2,'0')}</span><div>{text}<small>{claims.find(c=>c.path===`thesis.${i}`)?.status==='passage_matched'?'Inspect matching passage →':'Source evidence needed →'}</small></div></button>)}{!review.state.thesis?.length&&<p>No thesis statements recorded.</p>}</section>
        <section className="workspace-panel"><p className="workspace-eyebrow">02 / WHAT CHANGED</p><h3>Saved thesis comparison</h3>{prior?<div className="evidence-compare"><div><h4>Previous · {date(prior.createdAt)}</h4>{(prior.state.thesis||[]).map((t,i)=><p key={i}>{t}</p>)}</div><div><h4>Current · {date(review.createdAt)}</h4>{(review.state.thesis||[]).map((t,i)=><p key={i}>{t}</p>)}</div></div>:<p className="desk-explainer">No previous investment review exists. Changes cannot be independently compared yet.</p>}
        {prior&&(review.state.changes||[]).map((c,i)=><article className="evidence-change" key={i}><h4>{c.item}</h4><div className="evidence-compare"><p><small>Model-described prior</small>{c.prior||'Not specified'}</p><p><small>Model-described current</small>{c.current||'Not specified'}</p></div><p>{c.implication}</p><button onClick={()=>pick(`changes.${i}`)}>Inspect supporting evidence →</button></article>)}<p className="desk-explainer">These are saved review versions and model-described changes, not a verified event feed or amendments to your thesis.</p></section>
        <section className="workspace-panel"><p className="workspace-eyebrow">03 / FACT REGISTER</p><h3>Claims and provenance</h3>{claims.filter(c=>!c.path.startsWith('thesis.')&&!c.path.startsWith('changes.')).map(c=><button key={c.path} className="evidence-fact" onClick={()=>inspect(c)}><span>{c.type?.replaceAll('_',' ')||'Unclassified'}</span><strong>{c.statement}</strong><small>{c.status==='passage_matched'?'Passage matched':'Needs evidence'} →</small></button>)}{!claims.some(c=>!c.path.startsWith('thesis.')&&!c.path.startsWith('changes.'))&&<p className="desk-explainer">This review has no captured fact evidence. Older reports are not retroactively treated as verified.</p>}</section>
      </main><aside className="evidence-sidebar" aria-label="Quality and source evidence">
        <section className="workspace-panel"><p className="workspace-eyebrow">RESEARCH READINESS</p><h3>{review.quality.matchedCount} / {review.quality.claimCount} passages matched</h3><p className="desk-explainer">{review.quality.meaning}</p>{review.quality.issues.length>0?<ul>{review.quality.issues.map((issue,i)=><li key={i}>{issue}</li>)}</ul>:<p>No automated blockers recorded. Review assumptions and conclusions before use.</p>}<details><summary>Document coverage</summary><p>{review.documentsRead.length} read · {review.documentsNotRead.length} not read</p>{review.documentsRead.map((name,i)=><p key={i}>{name}</p>)}{review.documentsNotRead.map((name,i)=><p key={i}>Not read: {name}</p>)}</details></section>
        <section ref={inspector} className="workspace-panel evidence-source" aria-live="polite"><p className="workspace-eyebrow">SOURCE INSPECTOR</p><h3>{selected?'Supporting passages':'Inspect a claim'}</h3>{selected?<><p>{selected.statement}</p>{selected.evidence.length?selected.evidence.map((e,i)=>{const source=sources.find(s=>s.id===e.sourceId);return <article key={i}><strong>{source?.filename||'Unknown source'}</strong><small>{e.status==='passage_matched'?'Exact passage matched in captured extraction':'Passage could not be matched'}</small><blockquote>{e.excerpt||'No excerpt captured.'}</blockquote><p className="desk-explainer">{e.status==='passage_matched'?'A text match is not a conclusion check. Assess the surrounding context in the original document.':'Do not rely on this quotation until checked against the original.'}</p></article>}):<p>No source excerpt was captured. Open the company research to consult the originals.</p>}</>:<p className="desk-explainer">Select a thesis statement, change, or fact to inspect its source passage and matching status.</p>}</section>
      </aside></div>
    </>}
  </section>;
}
