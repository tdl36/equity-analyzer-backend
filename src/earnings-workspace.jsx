import * as React from 'react';
import {eventSources} from './event-sources.mjs';
import {eventResearch} from './earnings-model.mjs';
export function EarningsWorkspace({api,onRefresh,activities,onNavigate,onCompany,renderHtml}) {
  const [selected,setSelected]=React.useState(null),[query,setQuery]=React.useState(''),[filter,setFilter]=React.useState('all');
  const [instruction,setInstruction]=React.useState(''),[sending,setSending]=React.useState(false),[reply,setReply]=React.useState('');
  const [inventory,setInventory]=React.useState(null),[inventoryError,setInventoryError]=React.useState('');
  const sendLock=React.useRef(false);
  const events=eventResearch(activities);
  const visible=events.filter(e=>`${e.ticker} ${e.input.topic} ${e.analystName}`.toLowerCase().includes(query.toLowerCase())&&(filter==='all'||e.state===filter));
  const event=visible.find(e=>e.id===selected)||visible[0];
  const currentEvent=React.useRef(event?.id);currentEvent.current=event?.id;
  const labels={queued:'Awaiting synthesis',running:'Processing',failed:'Needs attention',draft:'Draft ready'};
  React.useEffect(()=>{setInstruction('');setReply('');},[event?.id]);
  React.useEffect(()=>{
    let active=true;let controller;setInventory(null);setInventoryError('');
    if(!event)return;
    const refresh=async()=>{controller?.abort();controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),15000);
      try{const r=await fetch(`${api}/api/agent/local-files/${encodeURIComponent(event.ticker)}`,{signal:controller.signal});if(!r.ok)throw Error('iCloud inventory unavailable');const value=await r.json();if(active){setInventory(value);setInventoryError('');}}
      catch(e){if(active)setInventoryError('Could not refresh the iCloud inventory. Any displayed inventory may be stale.');}finally{clearTimeout(timeout);}};
    refresh();const timer=setInterval(refresh,30000);return()=>{active=false;clearInterval(timer);controller?.abort();};
  },[api,event?.id,event?.ticker]);
  const liveSources=eventSources(inventory,event?.input?.topic,event?.sources||[]);
  const sendRevision=async()=>{
    if(!event||!instruction.trim()||sendLock.current||event.state==='running')return;
    const submittedEvent=event.id;
    sendLock.current=true;setSending(true);setReply('Submitting revision…');
    try {
      const res=await fetch(`${api}/api/analyst-activities/${encodeURIComponent(event.id)}/regenerate`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({customInstructions:instruction.trim()}),signal:AbortSignal.timeout(20000)});
      const body=await res.json();if(!res.ok)throw Error(body.error||'Revision could not be queued');
      if(currentEvent.current===submittedEvent){setReply('Revision queued with the covering analyst. The previous draft is preserved below. Review the new draft when processing finishes.');setInstruction('');}await onRefresh();
    }catch(e){if(currentEvent.current===submittedEvent)setReply(`Submission not confirmed: ${e.message}. Refresh the event before retrying to avoid duplicate work.`);await onRefresh();}
    finally{sendLock.current=false;setSending(false);}
  };
  return <section className="earnings-workspace">
    <header className="workspace-panel"><p className="workspace-eyebrow">EARNINGS / EVIDENCE / INVESTMENT IMPACT</p><h2>The event, in context.</h2><p className="desk-explainer">Review the source record, inspect the synthesis and compare it with your investment case. Active and recently failed analyst events appear here; approved events remain in the Analyst archive.</p>
      <div className="earnings-counts">{[['all','Events'],['draft','Drafts ready'],['running','Processing'],['failed','Need attention']].map(([id,label])=><button key={id} aria-pressed={filter===id} onClick={()=>setFilter(id)}><strong>{id==='all'?events.length:events.filter(e=>e.state===id).length}</strong>{label}</button>)}</div>
    </header>
    <div className="earnings-layout"><aside className="workspace-panel"><label className="earnings-search">Find an event<input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Ticker, event or analyst"/></label>
      <nav aria-label="Earnings events">{visible.map(e=><button className="earnings-event" key={e.id} aria-current={event?.id===e.id?'true':undefined} onClick={()=>setSelected(e.id)}><strong>{e.ticker} <small>{labels[e.state]}</small></strong><span>{e.input.topic}</span><small>{e.analystName||'Analyst team'}</small></button>)}</nav>{!visible.length&&<p className="workspace-empty">No events match this view. Create an event from a CATALYSTS folder in Analyst team.</p>}<button className="workspace-link-row" onClick={()=>onNavigate('analysts')}>Open Analyst team →</button></aside>
      {event&&<article className="workspace-panel earnings-detail" key={event.id}><p className="workspace-eyebrow">{event.ticker} / {labels[event.state]}</p><h2>{event.input.topic}</h2>
        <ol className="earnings-stages" aria-label="Evidence processing status"><li>Event detected</li><li>{event.sources.length?`${event.sources.length} source names recorded`:'Source record pending'}</li><li>{labels[event.state]}</li><li>Investment review pending</li></ol>
        {event.error&&<p className="workspace-error" role="alert">{String(event.error)}</p>}
        <section className="earnings-impact"><h3>Instruct {event.analystName||'the covering analyst'}</h3><p>Request a revision to this event’s recap—for example, “Reconcile the guidance figures and explain the extra-week effect.” This starts a source-based generation using configured API credits; it does not instantly edit or approve saved research.</p>
          <div role="log" aria-label="Revision instructions">{(event.output?.revisionInstructions||[]).map((m,i)=><p key={i}><strong>You:</strong> {m.content}</p>)}</div>
          <label className="earnings-search">Revision instructions<textarea rows={3} maxLength={6000} value={instruction} onChange={e=>setInstruction(e.target.value)} disabled={sending||event.state==='running'} placeholder="Tell the analyst what to correct, challenge or expand…"/></label>
          <button className="workspace-primary" disabled={sending||event.state==='running'||!instruction.trim()} onClick={sendRevision}>{sending?'Submitting…':event.state==='running'?'Analyst working…':'Send revision request'}</button><p role="status">{reply}</p>
          {!!event.output?.priorRuns?.length&&<details><summary>Previous drafts · {event.output.priorRuns.length}</summary>{event.output.priorRuns.map((r,i)=><details key={i}><summary>Draft {i+1} · {r.completedAt||'Date unavailable'}</summary><div className="desk-report-prose" dangerouslySetInnerHTML={{__html:renderHtml(r.synthesisMarkdown||'')}}/></details>)}</details>}
        </section>
        {event.output?.claimReview&&<section className="earnings-impact"><h3>What changed · evidence review</h3><p>Source quotations are checked by text matching. A separate model assesses support. These checks cover selected claims and remain subject to your review.</p>
          {(event.output.claimReview.changes||[]).map((c,i)=><div key={i}><h4>{c.area}: {c.change}</h4><p>{c.implication}</p><small>{c.baselineAvailable?'Comparison baseline supplied':'No verified prior baseline'} · Supporting claims: {c.claimIds.join(', ')||'None'} · Analyst review required</small></div>)}
          {!!event.output.claimReview.numericComparisons?.length&&<details open><summary>Numerical reconciliation · {event.output.claimReview.numericComparisons.length}</summary>{event.output.claimReview.numericComparisons.map((n,i)=><article className="amendment-card" key={i}><h4>{n.metric} · {n.status==='arithmetic_checked'?'Arithmetic checked':'Comparison needs review'}</h4><div className="evidence-compare"><div><strong>Actual / updated</strong><p>{n.actual?.value} {n.actual?.unit} · {n.actual?.period} · {n.actual?.basis}</p><details><summary>{n.actual?.filename||'Source unavailable'}</summary><blockquote>{n.actual?.quote}</blockquote></details></div><div><strong>{n.benchmarkType?.replaceAll('_',' ')||'Benchmark'}</strong><p>{n.benchmark?.value} {n.benchmark?.unit} · {n.benchmark?.period} · {n.benchmark?.basis}</p><details><summary>{n.benchmark?.filename||'Source unavailable'}</summary><blockquote>{n.benchmark?.quote}</blockquote></details></div></div>{n.delta!==null&&<p>Calculated difference: {n.delta} {n.deltaUnit}{n.relativePercent!==null?` · ${n.relativePercent}% relative change`:''}{n.basisPointDelta!==null?` · ${n.basisPointDelta} bps`:''}</p>}<ul>{(n.issues||[]).map((issue,k)=><li key={k}>{issue}</li>)}</ul><p>{n.limitation}</p></article>)}</details>}
          <details open><summary>Selected claim checks · {event.output.claimReview.claims?.length||0}</summary>{(event.output.claimReview.claims||[]).map(c=><details key={c.id}><summary>#{c.id} {c.passageMatched&&c.reviewPassed?'Passage matched · model check passed':'Needs evidence review'} — {c.statement}</summary><p>{c.kind} · {c.filename||'Unknown source'}{c.page?` · page ${c.page}`:''}</p><blockquote>{c.quote}</blockquote>{c.reviewIssue&&<p>{c.reviewIssue}</p>}</details>)}</details>
          <ul>{(event.output.claimReview.limitations||[]).map((l,i)=><li key={i}>{l}</li>)}</ul>
        </section>}
        <section className="earnings-impact"><h3>Event folder → current draft</h3>
          {inventoryError&&<p role="status">{inventoryError}</p>}
          {!liveSources.available?<p>The local agent has not provided an inventory for this view yet. This does not mean the event folder is empty.</p>:<><p>{liveSources.files.length} source files reported in CATALYSTS / {event.ticker} / {event.input.topic}. Last report: {liveSources.updated}. Inventory presence is not proof of readability.</p>
          <p>{event.draft?`${liveSources.added.length} filenames not recorded in this draft · ${liveSources.missing.length} recorded filenames no longer in the inventory`:'Draft source comparison will be available after synthesis.'}</p>
          <details><summary>Current event files</summary><ul>{liveSources.files.map((f,i)=><li key={i}>{f.path||f.filename} · {Math.round((f.size||0)/1024)} KB</li>)}</ul></details>
          <p>Changes to existing file contents are not detected by this filename comparison. Use the captured source hashes when auditing a specific draft.</p></>}
        </section>
        <h3>Source coverage</h3><p className="desk-explainer">Inferred from the filenames recorded with this output. “Not identified” does not prove a document is absent from iCloud. Broker reports and other sources are listed below.</p>
        <div className="earnings-coverage">{event.coverage.map(c=><div key={c.id}><strong>{c.label}</strong><span>{c.files.length?`${c.files.length} identified`:'Not identified'}</span></div>)}</div>
        <details><summary>Source register · {event.sources.length} named documents</summary>{event.sources.length?<ul>{event.sources.map(s=><li key={s}>{s}</li>)}</ul>:<p>No source filenames were attached to this activity.</p>}</details>
        {event.output?.evidenceSnapshot?.version===1&&<details open><summary>Recorded synthesis inputs · {event.output.evidenceSnapshot.sources?.length||0} documents</summary><p className="desk-explainer">Captured when this draft was generated. This records source delivery, not claim verification.</p><ul>{(event.output.evidenceSnapshot.sources||[]).map((s,i)=><li key={i}><strong>{s.filename}</strong> · {s.inputMode==='native_pdf'?'Native PDF':s.inputMode==='extracted_text'?'PDF text extraction':'Text'}{s.pages?` · ${s.pages} pages`:''}{s.characters!=null?` · ${s.characters.toLocaleString()} characters`:''}<br/><small>SHA-256: {s.sha256}</small></li>)}</ul></details>}
        <div className="earnings-checks"><h3>Evidence review</h3><ul><li>{event.sourceMismatch?`Coverage discrepancy: ${event.expected} documents reported, ${event.sources.length} filenames recorded.`:event.sources.length?'Source filenames available for review.':'Source completeness cannot be assessed yet.'}</li><li>{event.output?.claimReview?'A selected-claim source review is attached.':event.provenance?'A legacy model-generated source contribution record is attached.':'No source contribution record attached.'}</li><li>{event.output?.evidenceSnapshot?.version===1?'Synthesis input snapshot recorded.':'This recap predates input snapshots; source delivery has not been independently recorded.'}</li><li>Claim accuracy, page-level citations and numerical consistency are not independently verified for this recap.</li></ul></div>
        {event.provenance&&!event.output?.claimReview&&<details><summary>Source contribution record · model generated</summary><pre className="earnings-provenance">{typeof event.provenance==='string'?event.provenance:JSON.stringify(event.provenance,null,2)}</pre></details>}
        <section className="earnings-impact"><h3>Review the investment implications</h3><p>Compare the synthesis below with your saved thesis: what changed in earnings power, guidance, valuation, catalysts and downside risk? Separate reported facts from estimates and interpretation.</p><button className="workspace-link-row" onClick={()=>onCompany(event.ticker,'portfolio')}>Compare with {event.ticker} thesis →</button></section>
        <div className="workspace-section-heading"><h3>{event.state==='failed'&&event.draft?'Previous synthesis · latest attempt failed':'Event synthesis'}</h3><button onClick={()=>onNavigate('analysts')}>Revise or approve in Analyst team ↗</button></div>
        {event.draft?<div className="desk-report-prose" dangerouslySetInnerHTML={{__html:renderHtml(event.output.synthesisMarkdown)}}/>:<p className="workspace-empty">{event.state==='failed'?'Synthesis failed. Inspect the error and source folder before retrying in Analyst team.':event.state==='running'?'Synthesis is processing. This workspace refreshes with the research desk.':'No draft is attached yet. Open Analyst team to inspect or start synthesis.'}</p>}
      </article>}
    </div>
  </section>;
}
