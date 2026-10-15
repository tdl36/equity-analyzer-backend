import {emptyDraft,draftKey,readDraft,writeDraft} from './processing-proposal-draft.mjs';
import {ProcessingCoverage} from './processing-coverage';
import {ProcessingSourceLibrary} from './processing-source-library';
import * as React from 'react';
import {routeHash} from './workspace-model.mjs';
import {assignmentOutputs} from './assignment-defaults.mjs';

export function ProcessingProposals({api,job}){
 const [open,setOpen]=React.useState(false);
 return <details onToggle={e=>{if(e.currentTarget.open)setOpen(true);}}><summary>Choose sources &amp; prepare processing approval</summary>{open&&<ProposalReview key={api+':'+job.id} api={api} job={job}/>}</details>;
}
function ProposalReview({api,job}){
 const localKey=draftKey(api,job.id),[restored]=React.useState(()=>{try{return readDraft(sessionStorage,localKey);}catch{return {value:null,error:'Local draft storage is unavailable. Saved proposals are unaffected.'};}});
 const initial=restored.value||emptyDraft();
 const [draftIssue,setDraftIssue]=React.useState(restored.error),[restoredNotice,setRestoredNotice]=React.useState(!!restored.value);
 const [fullNote,setFullNote]=React.useState(initial.fullNote),[allowPdfText,setAllowPdfText]=React.useState(initial.allowPdfText),[meetingDate,setMeetingDate]=React.useState(initial.meetingDate),[meetingType,setMeetingType]=React.useState(initial.meetingType);
 const [supplements,setSupplements]=React.useState(initial.supplements);
 const [selected,setSelected]=React.useState(initial.selected),[purpose,setPurpose]=React.useState(initial.purpose),[title,setTitle]=React.useState(initial.title),[eventDate,setEventDate]=React.useState(initial.eventDate),[outputs,setOutputs]=React.useState(initial.outputs),[instruction,setInstruction]=React.useState(initial.instruction);
 const [rows,setRows]=React.useState([]),[busy,setBusy]=React.useState(false),[error,setError]=React.useState(''),[loaded,setLoaded]=React.useState(false),[pending,setPending]=React.useState(null);
 const lock=React.useRef(false),live=React.useRef(true);
 const authoring={version:1,selected,supplements,purpose,title,eventDate,outputs,instruction,fullNote,allowPdfText,meetingDate,meetingType};
 React.useLayoutEffect(()=>{if(draftIssue||pending)return;try{writeDraft(sessionStorage,localKey,authoring);}catch{setDraftIssue('Unsaved choices cannot be kept on this device. Save a proposal before leaving.');}},[localKey,JSON.stringify(authoring),draftIssue,pending]);
 const resetForm=()=>{const d=emptyDraft();setSelected(d.selected);setSupplements(d.supplements);setPurpose(d.purpose);setTitle(d.title);setEventDate(d.eventDate);setOutputs(d.outputs);setInstruction(d.instruction);setFullNote(d.fullNote);setAllowPdfText(d.allowPdfText);setMeetingDate(d.meetingDate);setMeetingType(d.meetingType);setRestoredNotice(false);try{sessionStorage.removeItem(localKey);setDraftIssue('');}catch{setDraftIssue('Local choices could not be cleared. Saved proposals are unaffected.');}};
 const key='charlie-processing-proposal:'+api+':'+job.id;
 const path='/api/research/assignments/'+job.id+'/processing-proposals';
 const call=async(url,body)=>{const response=await fetch(api+url,{...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(30000)});let data;try{data=await response.json();}catch{throw Error('Response unavailable. Refresh saved proposals before retrying.');}if(!response.ok){const e=Error(data.error||'Processing proposals unavailable');e.status=response.status;throw e;}return data;};
 const refresh=async()=>{const data=await call(path);if(!Array.isArray(data.proposals))throw Error('Proposal list unavailable. No processing was requested.');if(live.current){setRows(data.proposals);setLoaded(true);}const raw=sessionStorage.getItem(key);if(raw){const saved=JSON.parse(raw);if(data.proposals.some(p=>p.id===saved.requestId)){sessionStorage.removeItem(key);if(live.current){setPending(null);resetForm();}}}};
 React.useEffect(()=>{live.current=true;try{const raw=sessionStorage.getItem(key);if(raw)setPending(JSON.parse(raw));}catch{setError('Saved submission could not be read. Inspect saved proposals before creating another.');return;}refresh().catch(e=>live.current&&setError(e.message));return()=>{live.current=false;};},[key]);
 const act=async fn=>{if(lock.current)return;lock.current=true;setBusy(true);setError('');try{await fn();await refresh();}catch(e){if(live.current)setError(e.message);}finally{lock.current=false;if(live.current)setBusy(false);}};
 const save=()=>act(async()=>{
  const body=pending||{requestId:crypto.randomUUID(),filenames:[...selected].sort(),supplementalIds:[...supplements].sort(),plan:{researchPurpose:purpose==='event_note'?{kind:purpose,event:{title:title.trim(),date:eventDate}}:{kind:purpose},outputs:[...outputs].sort(),instruction,horizon:'12–24 months',...(fullNote?{labFullNote:{allowPdfText}}:{}),...(outputs.includes('meeting_prep')?{meeting:{date:meetingDate,type:meetingType}}:{})}};
  sessionStorage.setItem(key,JSON.stringify(body));setPending(body);
  try{const receipt=await call(path,body);if(receipt.id!==body.requestId)throw Error('Proposal receipt mismatch. Refresh saved proposals.');sessionStorage.removeItem(key);setPending(null);resetForm();}
  catch(e){if(e.status===409){sessionStorage.removeItem(key);setPending(null);}throw e;}
 });
 return <section aria-label="Processing approval" style={{padding:'12px 0',overflowWrap:'anywhere'}}>
  <p><strong>Collected library → selected evidence → your processing approval</strong></p>
  <p>Collection searched {job.input.since} through {job.input.until}. This is collection coverage, not proof that the accepted thesis incorporates these documents. Saving a proposal spends no research credits.</p>
  {error&&<p role="alert">{error}</p>}
  {restoredNotice&&<p role="status">Restored unsaved choices in this browser tab. Review them; originals and baseline are checked again when you save. No processing was requested.</p>}
  {draftIssue&&<p role="alert">{draftIssue}</p>}
  <p><small>Unsaved choices are kept in this browser tab while storage is available. Saving a proposal creates the durable server record.</small> <button disabled={busy||!!pending} onClick={resetForm}>Discard unsaved choices</button></p>
  <button disabled={busy} onClick={()=>act(async()=>{})}>Refresh saved proposals</button>
  {pending&&<p role="status">A proposal submission is unresolved. Refresh to find its receipt, or retry the same saved submission below. Its choices are frozen.</p>}
  <fieldset disabled={busy||!loaded||!!pending}>
   <legend>Select evidence and purpose</legend>
   <label>Processing purpose<select aria-label="Processing purpose" value={purpose} onChange={e=>{setPurpose(e.target.value);setOutputs(e.target.value==='event_note'?['summary_note']:['thesis']);}}><option value="thesis_review">Thesis review · longitudinal</option><option value="event_note">Event note · specific catalyst</option></select></label>
   <p>{purpose==='thesis_review'?'Review the full investment case over time. Add relevant foundational and historical originals from the wider library below. The collection window does not limit the investment case; missing evidence remains a gap.':'Debrief an identified event with contextual history. This does not change the thesis or mark it current. Thesis implications need a separate proposal.'}</p>
   {purpose==='event_note'&&<div className="sa-form-grid"><label>Event title<input aria-label="Processing event title" maxLength={200} value={title} onChange={e=>setTitle(e.target.value)}/></label><label>Event date<input aria-label="Processing event date" type="date" value={eventDate} onChange={e=>setEventDate(e.target.value)}/></label></div>}
   <p>{selected.length} of {job.result.sources.length} collected originals selected, plus {supplements.length} supplemental originals · none selected automatically. Select at least one collected original to link this proposal to the collection. Up to 15 MB per original and 60 MB per processing pack; collection itself can be larger.</p>
   <div style={{maxHeight:280,overflow:'auto',border:'1px solid var(--border-color,#d8dedd)',padding:12}}>{job.result.sources.map(s=><label className="sa-check" key={s.filename} style={{display:'flex',alignItems:'start',gap:8,padding:'6px 0'}}><input type="checkbox" checked={selected.includes(s.filename)} disabled={!selected.includes(s.filename)&&selected.length+supplements.length>=1000} onChange={e=>setSelected(p=>e.target.checked?[...p,s.filename]:p.filter(n=>n!==s.filename))}/><span>{s.filename}</span></label>)}</div>
   <ProcessingSourceLibrary call={call} path={path.replace('processing-proposals','processing-library')} selected={supplements} onChange={setSupplements} purpose={purpose} remaining={1000-selected.length}/>
   <div className="sa-form-grid">{assignmentOutputs.filter(([id])=>purpose!=='event_note'||id==='summary_note'||id==='meeting_prep').map(([id,label])=><label className="sa-check" key={id}><input type="checkbox" checked={outputs.includes(id)} onChange={e=>setOutputs(p=>e.target.checked?[...p,id]:p.filter(n=>n!==id))}/>{label}</label>)}</div>
   {outputs.includes('meeting_prep')&&<div className="sa-form-grid"><label>Meeting date<input aria-label="Processing meeting date" type="date" value={meetingDate} onChange={e=>setMeetingDate(e.target.value)}/></label><label>Meeting type<select aria-label="Processing meeting type" value={meetingType} onChange={e=>setMeetingType(e.target.value)}>{['management','expert','conference','other'].map(t=><option key={t} value={t}>{t}</option>)}</select></label><p>Meeting questions use the shared research report; no additional generation stage.</p></div>}
   <label className="sa-check"><input type="checkbox" checked={fullNote} onChange={e=>setFullNote(e.target.checked)}/>Full versioned Lab note · additional research credits</label>
   {fullNote&&<div><p>Saving checks source readability and batch limits without model calls, then freezes the model, instructions and source representations. The full note uses up to eight additional model calls, capped at 32,000 output tokens per call. Large evidence packs are processed in bounded stages; omissions remain visible.</p><label className="sa-check"><input type="checkbox" checked={allowPdfText} onChange={e=>setAllowPdfText(e.target.checked)}/>Allow extracted PDF text when needed · charts and layout may be omitted</label></div>}
   <label>Focus and evidence gaps<textarea rows={3} maxLength={1800} value={instruction} onChange={e=>setInstruction(e.target.value)}/></label>
  </fieldset>
  <button disabled={busy||!loaded||(!pending&&(!selected.length||!outputs.length||(outputs.includes('meeting_prep')&&!meetingDate)||(purpose==='event_note'&&(!title.trim()||!eventDate))))} onClick={save}>{pending?'Retry saved proposal submission':'Save proposal · no generation'}</button>
  <p><small>Approval starts the existing shared research workflow: up to twelve model calls within the configured budget. Large evidence packs may be excerpted with omissions recorded; selection does not guarantee exhaustive review. No exact credit estimate is available. Finished research still requires investor review.</small></p>
  {!loaded?<p>Loading saved proposals…</p>:!rows.length?<p>No saved processing proposals for this collection.</p>:rows.map(p=><article key={p.id} style={{borderTop:'1px solid var(--border-color,#d8dedd)',padding:'14px 0'}}>
   <strong>{p.plan.researchPurpose.kind==='event_note'?p.plan.researchPurpose.event.title:'Longitudinal thesis review'}</strong> · {p.status}
   <p>{p.plan.researchPurpose.kind==='event_note'?`Event ${p.plan.researchPurpose.event.date} · `:''}{p.sources.length} verified originals at save · {p.plan.outputs.map(id=>assignmentOutputs.find(x=>x[0]===id)?.[1]||id).join(', ')}</p>
   {p.plan.meeting&&<p>Meeting preparation · {p.plan.meeting.date} · {p.plan.meeting.type}</p>}
   {p.fullNote&&<p>Full versioned note · frozen model {p.fullNote.modelKey} · up to {p.fullNote.maxCalls} additional calls, {p.fullNote.maxTokens.toLocaleString()} output tokens per call · PDF text fallback {p.fullNote.allowPdfText?'allowed':'off'}. Draft requires investor review.</p>}
   {p.fullNote?.sourcePreparation&&<div><p>Full-note source check: {p.fullNote.sourcePreparation.sourcePack.length} originals planned across {p.fullNote.sourcePreparation.batchCount} batches. Rechecked before dispatch; provider context validation still happens at execution.</p>{p.fullNote.sourcePreparation.warnings.length>0&&<details><summary>Source representation limitations · {p.fullNote.sourcePreparation.warnings.length}</summary><ul>{p.fullNote.sourcePreparation.warnings.map(w=><li key={w.reference}>{p.sources.find(s=>s.reference===w.reference)?.filename||w.reference}: {w.issue}</li>)}</ul></details>}</div>}
   {p.plan.instruction&&<p>{p.plan.instruction}</p>}
   <ProcessingCoverage coverage={p.coverage}/>
   <details><summary>Frozen sources and scope</summary><p>Revision {p.revision}. Collection {p.plan.since}–{p.plan.until}. Sources and baseline will be checked again before dispatch.</p>{p.fullNote&&<p>Full-note configuration: {p.fullNote.configDigest}<br/>Note instructions: {p.fullNote.promptSha256}</p>}<ul>{p.sources.map(s=><li key={s.documentId}><a href={api+'/api/pipeline-lab/original/'+encodeURIComponent(p.ticker)+'/thesis_'+s.documentId} target="_blank" rel="noreferrer">{s.filename}</a> · {(s.bytes/1000000).toFixed(2)} MB{p.sourceContext?.[s.documentId]&&<p><small>{p.sourceContext[s.documentId].role==='supplemental'?'Supplemental history; not collected by this request':'From collection receipt'} · Published: {p.sourceContext[s.documentId].publishedDate||'unknown'} · {p.sourceContext[s.documentId].referencedBySavedThesis?'Exact bytes referenced by saved thesis; incorporation still needs review':'No exact-hash reference found in saved thesis register; incorporation unknown'}</small></p>}</li>)}</ul></details>
   {p.status==='pending'&&<><button className="workspace-primary" disabled={busy} onClick={()=>act(async()=>{await call('/api/research/processing-proposals/'+p.id+'/approve',{revision:p.revision,digest:p.digest});})}>Approve processing · uses research credits</button> <button disabled={busy} onClick={()=>act(async()=>{await call('/api/research/processing-proposals/'+p.id+'/cancel',{revision:p.revision,digest:p.digest});})}>Cancel proposal</button></>}
   {p.assignmentId&&<p><a href={routeHash('stockanalysis',p.ticker,'stock-analysis','assignment:'+p.assignmentId)}>Open saved processing job and outputs</a></p>}
  </article>)}
 </section>;
}
