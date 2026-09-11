import * as React from 'react';
const fields={support:'Supporting evidence',contrary:'Risks & contrary evidence',nextTest:'Next test'};
const ready=c=>c.passageMatched&&c.reviewPassed;
export function ProposalReview({job,body,revision,disabled,onCommit,onDebate}){
 const changes=job.result?.changes||[];
 const [filter,setFilter]=React.useState('ready'),[selectedId,setSelectedId]=React.useState(null),[assumptionId,setAssumptionId]=React.useState(''),[field,setField]=React.useState('support');
 const available=changes.filter(c=>filter==='ready'?ready(c):!ready(c));
 const selected=available.find(c=>c.id===selectedId)||available[0];
 const assumption=body.assumptions.find(a=>a.id===(selected?.assumptionId||assumptionId));
 const effectiveField=selected?.field||field;
 const applied=!!selected&&!!assumption&&assumption[effectiveField]===selected.after;
 const canAccept=selected&&ready(selected)&&['awaiting_approval','applied'].includes(job.status)&&assumption&&!applied;
 const title=c=>body.assumptions.find(a=>a.id===c.assumptionId)?.claim||'Research change';
 return <div className="proposal-review"><div className="proposal-review-filters" aria-label="Filter proposed changes">{[['ready','Ready to review',changes.filter(ready).length],['flagged','Needs verification',changes.filter(c=>!ready(c)).length]].map(([id,label,count])=><button key={id} aria-pressed={filter===id} onClick={()=>{setFilter(id);setSelectedId(null);}}>{label}<strong>{count}</strong></button>)}</div>
 <p className="proposal-review-intro">{filter==='ready'?'These changes passed source-passage and model checks. Read each proposal before accepting it.':'These changes are blocked from acceptance. Inspect a card to see the proposed wording and the verification problem.'}</p>
 {!available.length?<p className="case-evidence-banner">{filter==='ready'?'No changes are ready for acceptance. Open Needs verification to inspect any flagged drafts.':'No changes need additional verification.'}</p>:<div className="proposal-review-layout"><nav className="proposal-change-list" aria-label="Proposed changes">{available.map((c,i)=><button key={c.id} aria-pressed={selected?.id===c.id} onClick={()=>{setSelectedId(c.id);setAssumptionId('');setField('support');}}><small>CHANGE {changes.indexOf(c)+1} · {fields[c.field]||'Research wording'}</small><span>{title(c)}</span><em>{body.assumptions.some(a=>a.id===c.assumptionId&&a[c.field]===c.after)?'Already reflected in case':ready(c)?'Open for review →':'Verification required →'}</em></button>)}</nav>
 {selected&&<article className="proposal-change-reader" aria-label="Selected change"><header><p className="workspace-eyebrow">CHANGE {changes.indexOf(selected)+1} / {fields[effectiveField]||'PROPOSED WORDING'}</p><h3>{title(selected)}</h3><span className="proposal-status">{applied?'Already reflected in current case':ready(selected)?'Ready for your review':'Blocked · needs verification'}</span></header>
 <section><h4>Why Charlie suggests this</h4><p>{selected.reason||'No rationale recorded.'}</p></section>
 {!ready(selected)&&<section className="proposal-verification"><h4>What needs verification</h4><p>{!selected.passageMatched?'The supporting quotation did not match the saved source. ':''}{selected.reviewIssue||(!selected.reviewPassed?'The independent review did not approve this wording.':'')}</p></section>}
 <details className="proposal-before"><summary>Compare with current wording</summary><p>{assumption?.[effectiveField]||selected.before||'Not recorded'}</p></details>
 <section className="proposal-wording"><h4>Proposed replacement</h4><p>{selected.after}</p></section>
 <details className="proposal-source"><summary>Read supporting source passages · {selected.evidence?.length||0}</summary>{selected.evidence?.map((e,i)=><section key={i}><h4>{job.result.sources?.find(s=>s.id===e.sourceId)?.filename||'Source reference unavailable'}</h4><small>{e.status==='passage_matched'?'Passage matched at generation':'Passage not verified'}</small><blockquote>{e.excerpt||'No excerpt available'}</blockquote></section>)}</details>
 <div className="proposal-actions"><button onClick={()=>onDebate({prompt:'Challenge this proposed interpretation. Separate what the source actually says from inference, address verification findings, and identify missing evidence. Do not claim any edits were applied.',content:JSON.stringify({caseRevision:revision,investmentCase:body,proposalId:job.id,change:selected,sources:job.result.sources})})}>Discuss with Charlie</button>
 {ready(selected)&&!selected.assumptionId&&<><label>Investment assumption<select value={assumptionId} onChange={e=>setAssumptionId(e.target.value)}><option value="">Choose an assumption</option>{body.assumptions.map(a=><option key={a.id} value={a.id}>{a.claim}</option>)}</select></label><label>Destination<select value={field} onChange={e=>setField(e.target.value)}>{Object.entries(fields).map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label></>}
 {ready(selected)?<button className="workspace-primary" disabled={disabled||!canAccept} onClick={()=>onCommit({mode:'source_change',sourceChange:{jobId:job.id,changeId:selected.id,assumptionId:assumption.id,field:effectiveField}})}>{applied?'Already reflected in case':'Accept this change'}</button>:<p>Acceptance is unavailable until the verification issues are resolved in a new or repaired proposal.</p>}</div>
 <p className="proposal-footnote">Accepting saves a new case revision. Discussion and opening a card do not change your research.</p>
 </article>}</div>}
 </div>;
}
