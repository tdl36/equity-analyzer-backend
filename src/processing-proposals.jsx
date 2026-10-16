import {processingSelection} from './processing-selection-model.mjs';
import {ProcessingSelectionReview} from './processing-selection-review';
import {OriginalDownload} from './original-download';
import {CollectedSourcePicker} from './collected-source-picker';
import {processingProposal,processingProposalPage,mergeProcessingProposals} from './processing-proposal-status.mjs';
import {emptyDraft,emptyPurposeDraft,draftKey,readDraftWorkspace,writeDraftWorkspace,readSubmission} from './processing-proposal-draft.mjs';
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
 const localKey=draftKey(api,job.id),[restored]=React.useState(()=>{try{return readDraftWorkspace(sessionStorage,localKey);}catch{return {value:null,error:'Local draft storage is unavailable. Saved proposals are unaffected.'};}});
 const initial=restored.value||emptyDraft();
 const purposeDrafts=React.useRef(restored.drafts||{});
 const [submissionIssue,setSubmissionIssue]=React.useState('');
 const [draftIssue,setDraftIssue]=React.useState(restored.error),[restoredNotice,setRestoredNotice]=React.useState(!!restored.value);
 const [fullNote,setFullNote]=React.useState(initial.fullNote),[allowPdfText,setAllowPdfText]=React.useState(initial.allowPdfText),[meetingDate,setMeetingDate]=React.useState(initial.meetingDate),[meetingType,setMeetingType]=React.useState(initial.meetingType);
 const [collectionInventory,setCollectionInventory]=React.useState({rows:[],loaded:false}),[libraryInventory,setLibraryInventory]=React.useState({rows:[],loaded:false});
 const [collectionReady,setCollectionReady]=React.useState(false);
 const [supplements,setSupplements]=React.useState(initial.supplements);
 const [selected,setSelected]=React.useState(initial.selected),[purpose,setPurpose]=React.useState(initial.purpose),[title,setTitle]=React.useState(initial.title),[eventDate,setEventDate]=React.useState(initial.eventDate),[outputs,setOutputs]=React.useState(initial.outputs),[instruction,setInstruction]=React.useState(initial.instruction);
 const [rows,setRows]=React.useState([]),[busy,setBusy]=React.useState(false),[error,setError]=React.useState(''),[readIssue,setReadIssue]=React.useState(''),[loaded,setLoaded]=React.useState(false),[pending,setPending]=React.useState(null);
 const [nextCursor,setNextCursor]=React.useState(null),[historyIssue,setHistoryIssue]=React.useState('');
 const readEpoch=React.useRef(0);
 const lock=React.useRef(false),live=React.useRef(true);
 const selectionSummary=processingSelection({selected,supplements,collection:collectionInventory,library:libraryInventory,since:job.input.since,until:job.input.until});
 const authoring={version:1,selected,supplements,purpose,title,eventDate,outputs,instruction,fullNote,allowPdfText,meetingDate,meetingType};
 React.useLayoutEffect(()=>{if(draftIssue||pending)return;try{writeDraftWorkspace(sessionStorage,localKey,authoring,purposeDrafts.current);purposeDrafts.current[purpose]=authoring;}catch{setDraftIssue('Unsaved choices cannot be kept on this device. Save a proposal before leaving.');}},[localKey,JSON.stringify(authoring),draftIssue,pending]);
 const applyDraft=d=>{setSelected(d.selected);setSupplements(d.supplements);setPurpose(d.purpose);setTitle(d.title);setEventDate(d.eventDate);setOutputs(d.outputs);setInstruction(d.instruction);setFullNote(d.fullNote);setAllowPdfText(d.allowPdfText);setMeetingDate(d.meetingDate);setMeetingType(d.meetingType);};
 const changePurpose=next=>{purposeDrafts.current[purpose]=authoring;applyDraft(purposeDrafts.current[next]||emptyPurposeDraft(next));setRestoredNotice(false);};
 const resetForm=()=>{const d=emptyPurposeDraft(purpose);purposeDrafts.current[purpose]=d;applyDraft(d);setRestoredNotice(false);try{writeDraftWorkspace(sessionStorage,localKey,d,purposeDrafts.current);setDraftIssue('');}catch{setDraftIssue('Current draft could not be cleared in browser recovery. Keep this screen open.');}};
 const key='charlie-processing-proposal:'+api+':'+job.id;
 const path='/api/research/assignments/'+job.id+'/processing-proposals';
 const call=async(url,body)=>{const response=await fetch(api+url,{...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(30000)});let data;try{data=await response.json();}catch{throw Error('Response unavailable. Refresh saved proposals before retrying.');}if(!response.ok){const e=Error(data.error||'Processing proposals unavailable');e.status=response.status;throw e;}return data;};
 const refresh=async()=>{const epoch=++readEpoch.current;const current=()=>live.current&&epoch===readEpoch.current;try{
  const data=await call(path),page=processingProposalPage(data,job);let proposals=page.rows;
  if(!current())return;
  if(current()){setRows(proposals);setNextCursor(page.nextCursor);setLoaded(true);}
  // Browser recovery must not hide verified server records.
  let recovered;try{recovered=readSubmission(sessionStorage,key);}catch{recovered=readSubmission(null,key);}
  let issue=recovered.error;
  if(recovered.value){
   if(live.current)setPending(recovered.value);
   if(!proposals.some(p=>p.id===recovered.value.requestId)){
    const exact=processingProposalPage(await call(path+'?requestId='+encodeURIComponent(recovered.value.requestId)),job,{requestId:recovered.value.requestId});
    proposals=mergeProcessingProposals(proposals,exact.rows);
   }
   if(!current())return;
   if(proposals.some(p=>p.id===recovered.value.requestId)){
    try{sessionStorage.removeItem(key);if(live.current){setPending(null);resetForm();}}
    catch{issue='The saved proposal was found, but its local recovery record could not be cleared. Restore browser storage and refresh before submitting again.';}
   }else if(live.current)setPending(recovered.value);
  }else if(!issue&&live.current)setPending(null);
  if(current()){setSubmissionIssue(issue);setRows(proposals);setNextCursor(page.nextCursor);setHistoryIssue('');setLoaded(true);setReadIssue('');}
 }catch(e){if(!current())return;setReadIssue(e.message);throw e;}};
 const older=async()=>{if(lock.current||!nextCursor)return;lock.current=true;setBusy(true);setHistoryIssue('');const cursor=nextCursor,epoch=readEpoch.current;try{
  const page=processingProposalPage(await call(path+'?after='+encodeURIComponent(cursor)),job,{after:cursor});
  if(live.current&&epoch===readEpoch.current){setRows(old=>mergeProcessingProposals(old,page.rows));setNextCursor(page.nextCursor);}
 }catch(e){if(live.current&&epoch===readEpoch.current)setHistoryIssue(e.message);}finally{lock.current=false;if(live.current)setBusy(false);}};
 React.useEffect(()=>{live.current=true;refresh().catch(e=>live.current&&setError(e.message));return()=>{live.current=false;readEpoch.current++;};},[key]);
 const act=async fn=>{if(lock.current)return;lock.current=true;setBusy(true);setError('');try{await fn();await refresh();}catch(e){if(live.current)setError(e.message);}finally{lock.current=false;if(live.current)setBusy(false);}};
 const save=()=>act(async()=>{
  if(submissionIssue)throw Error(submissionIssue);
  const body=pending||{requestId:crypto.randomUUID(),filenames:[...selected].sort(),supplementalIds:[...supplements].sort(),plan:{researchPurpose:purpose==='event_note'?{kind:purpose,event:{title:title.trim(),date:eventDate}}:{kind:purpose},outputs:[...outputs].sort(),instruction,horizon:'12–24 months',...(fullNote?{labFullNote:{allowPdfText}}:{}),...(outputs.includes('meeting_prep')?{meeting:{date:meetingDate,type:meetingType}}:{})}};
  try{sessionStorage.setItem(key,JSON.stringify(body));}catch{setSubmissionIssue('Submission was not sent because browser recovery could not be saved. Restore browser storage and refresh before submitting.');throw Error('No proposal was sent. Browser recovery storage is unavailable.');}
  setPending(body);
  try{const receipt=processingProposal(await call(path,body),job);if(receipt.id!==body.requestId)throw Error('Proposal receipt mismatch. Refresh saved proposals.');sessionStorage.removeItem(key);setPending(null);resetForm();}
  catch(e){if(e.status===409){sessionStorage.removeItem(key);setPending(null);}throw e;}
 });
 return <section aria-label="Processing approval" style={{padding:'12px 0',overflowWrap:'anywhere'}}>
  <p><strong>Collected library → selected evidence → your processing approval</strong></p>
  <p>Collection searched {job.input.since} through {job.input.until}. This is collection coverage, not proof that the accepted thesis incorporates these documents. Saving a proposal spends no research credits.</p>
  {readIssue&&<p role="alert">{readIssue}</p>}{error&&error!==readIssue&&<p role="alert">{error}</p>}
  {restoredNotice&&<p role="status">Restored unsaved choices in this browser tab. Review them; originals and baseline are checked again when you save. No processing was requested.</p>}
  {draftIssue&&<p role="alert">{draftIssue}</p>}
  {submissionIssue&&<p role="alert">{submissionIssue}</p>}
  <p><small>Unsaved choices are kept in this browser tab while storage is available. Saving a proposal creates the durable server record.</small> <button disabled={busy||!!pending} onClick={resetForm}>Discard current purpose draft</button></p>
  <button disabled={busy} onClick={()=>act(async()=>{})}>Refresh saved proposals</button>
  {pending&&<p role="status">A proposal submission is unresolved. Refresh to find its receipt, or retry the same saved submission below. Its choices are frozen.</p>}
  <fieldset disabled={busy||!loaded||!!pending||!!readIssue}>
   <legend>Select evidence and purpose</legend>
   <label>Processing purpose<select aria-label="Processing purpose" value={purpose} onChange={e=>changePurpose(e.target.value)}><option value="thesis_review">Thesis review · longitudinal</option><option value="event_note">Event note · specific catalyst</option></select></label>
   <p><small>Each purpose keeps its own unfinished sources, outputs and instructions in this browser tab. Switching restores that purpose’s draft; it never copies or submits the other one.</small></p>
   <p>{purpose==='thesis_review'?'Review the full investment case over time. Add relevant foundational and historical originals from the wider library below. The collection window does not limit the investment case; missing evidence remains a gap.':'Debrief an identified event with contextual history. This does not change the thesis or mark it current. Thesis implications need a separate proposal.'}</p>
   {purpose==='event_note'&&<div className="sa-form-grid"><label>Event title<input aria-label="Processing event title" maxLength={200} value={title} onChange={e=>setTitle(e.target.value)}/></label><label>Event date<input aria-label="Processing event date" type="date" value={eventDate} onChange={e=>setEventDate(e.target.value)}/></label></div>}
   <p>{selected.length} of {job.result.sources.length} collected originals selected, plus {supplements.length} supplemental originals · none selected automatically. Select at least one collected original to link this proposal to the collection. Up to 15 MB per original and 60 MB per processing pack; collection itself can be larger.</p>
   <CollectedSourcePicker api={api} call={call} job={job} selected={selected} onChange={setSelected} purpose={purpose} remaining={1000-supplements.length} onReady={setCollectionReady} onInventory={setCollectionInventory}/>
   <ProcessingSourceLibrary api={api} collectionId={job.id} ticker={job.ticker} call={call} path={path.replace('processing-proposals','processing-library')} selected={supplements} onChange={setSupplements} purpose={purpose} remaining={1000-selected.length} onInventory={setLibraryInventory}/>
   <ProcessingSelectionReview summary={selectionSummary} purpose={purpose} onRemove={r=>r.role==='collection'?setSelected(selected.filter(n=>n!==r.selection)):setSupplements(supplements.filter(id=>id!==r.selection))}/>
   <div className="sa-form-grid">{assignmentOutputs.filter(([id])=>purpose!=='event_note'||id==='summary_note'||id==='meeting_prep').map(([id,label])=><label className="sa-check" key={id}><input type="checkbox" checked={outputs.includes(id)} onChange={e=>setOutputs(p=>e.target.checked?[...p,id]:p.filter(n=>n!==id))}/>{label}</label>)}</div>
   {outputs.includes('meeting_prep')&&<div className="sa-form-grid"><label>Meeting date<input aria-label="Processing meeting date" type="date" value={meetingDate} onChange={e=>setMeetingDate(e.target.value)}/></label><label>Meeting type<select aria-label="Processing meeting type" value={meetingType} onChange={e=>setMeetingType(e.target.value)}>{['management','expert','conference','other'].map(t=><option key={t} value={t}>{t}</option>)}</select></label><p>Meeting questions use the shared research report; no additional generation stage.</p></div>}
   <label className="sa-check"><input type="checkbox" checked={fullNote} onChange={e=>setFullNote(e.target.checked)}/>Full versioned Lab note · additional research credits</label>
   {fullNote&&<div><p>Saving checks source readability and batch limits without model calls, then freezes the model, instructions and source representations. The full note uses up to eight additional model calls, capped at 32,000 output tokens per call. Large evidence packs are processed in bounded stages; omissions remain visible.</p><label className="sa-check"><input type="checkbox" checked={allowPdfText} onChange={e=>setAllowPdfText(e.target.checked)}/>Allow extracted PDF text when needed · charts and layout may be omitted</label></div>}
   <label>Focus and evidence gaps<textarea rows={3} maxLength={1800} value={instruction} onChange={e=>setInstruction(e.target.value)}/></label>
  </fieldset>
  <button disabled={busy||!loaded||!!readIssue||!!submissionIssue||(!pending&&(!collectionReady||selectionSummary.metadataPending||selectionSummary.missing>0||selectionSummary.unavailable>0||!selected.length||!outputs.length||(outputs.includes('meeting_prep')&&!meetingDate)||(purpose==='event_note'&&(!title.trim()||!eventDate))))} onClick={save}>{pending?'Retry saved proposal submission':'Save proposal · no generation'}</button>
  <p><small>Approval starts the existing shared research workflow: up to twelve model calls within the configured budget. Large evidence packs may be excerpted with omissions recorded; selection does not guarantee exhaustive review. No exact credit estimate is available. Finished research still requires investor review.</small></p>
  {!loaded?<p>{readIssue?'Saved proposals unavailable. Refresh to check their status.':'Loading saved proposals…'}</p>:!rows.length?<p>No saved processing proposals for this collection.</p>:rows.map(p=><article key={p.id} style={{borderTop:'1px solid var(--border-color,#d8dedd)',padding:'14px 0'}}>
   <strong>{p.plan.researchPurpose.kind==='event_note'?p.plan.researchPurpose.event.title:'Longitudinal thesis review'}</strong> · {p.status}
   <p>{p.plan.researchPurpose.kind==='event_note'?`Event ${p.plan.researchPurpose.event.date} · `:''}{p.sources.length} verified originals at save · {p.plan.outputs.map(id=>assignmentOutputs.find(x=>x[0]===id)?.[1]||id).join(', ')}</p>
   {p.plan.meeting&&<p>Meeting preparation · {p.plan.meeting.date} · {p.plan.meeting.type}</p>}
   {p.fullNote&&<p>Full versioned note · frozen model {p.fullNote.modelKey} · up to {p.fullNote.maxCalls} additional calls, {p.fullNote.maxTokens.toLocaleString()} output tokens per call · PDF text fallback {p.fullNote.allowPdfText?'allowed':'off'}. Draft requires investor review.</p>}
   {p.fullNote?.sourcePreparation&&<div><p>Full-note source check: {p.fullNote.sourcePreparation.sourcePack.length} originals planned across {p.fullNote.sourcePreparation.batchCount} batches. Rechecked before dispatch; provider context validation still happens at execution.</p>{p.fullNote.sourcePreparation.warnings.length>0&&<details><summary>Source representation limitations · {p.fullNote.sourcePreparation.warnings.length}</summary><ul>{p.fullNote.sourcePreparation.warnings.map(w=><li key={w.reference}>{p.sources.find(s=>s.reference===w.reference)?.filename||w.reference}: {w.issue}</li>)}</ul></details>}</div>}
   {p.plan.instruction&&<p>{p.plan.instruction}</p>}
   <ProcessingCoverage coverage={p.coverage}/>
   <details><summary>Frozen sources and scope</summary><p>Revision {p.revision}. Collection {p.plan.since}–{p.plan.until}. Sources and baseline will be checked again before dispatch.</p>{p.fullNote&&<p>Full-note configuration: {p.fullNote.configDigest}<br/>Note instructions: {p.fullNote.promptSha256}</p>}<ul>{p.sources.map(s=><li key={s.documentId}>{s.filename} <OriginalDownload url={api+'/api/pipeline-lab/original/'+encodeURIComponent(p.ticker)+'/thesis_'+s.documentId} filename={s.filename} sha256={s.sha256}/> · {(s.bytes/1000000).toFixed(2)} MB{p.sourceContext?.[s.documentId]&&<p><small>{p.sourceContext[s.documentId].role==='supplemental'?'Supplemental history; not collected by this request':'From collection receipt'} · Published: {p.sourceContext[s.documentId].publishedDate||'unknown'} · {p.sourceContext[s.documentId].referencedBySavedThesis?'Exact bytes referenced by saved thesis; incorporation still needs review':'No exact-hash reference found in saved thesis register; incorporation unknown'}</small></p>}</li>)}</ul></details>
   {p.status==='pending'&&<><button className="workspace-primary" disabled={busy||!!readIssue||!loaded} onClick={()=>act(async()=>{await call('/api/research/processing-proposals/'+p.id+'/approve',{revision:p.revision,digest:p.digest});})}>Approve processing · uses research credits</button> <button disabled={busy||!!readIssue||!loaded} onClick={()=>act(async()=>{await call('/api/research/processing-proposals/'+p.id+'/cancel',{revision:p.revision,digest:p.digest});})}>Cancel proposal</button></>}
   {p.assignmentId&&<p><a href={routeHash('stockanalysis',p.ticker,'stock-analysis','assignment:'+p.assignmentId)}>Open saved processing job and outputs</a></p>}
  </article>)}
  {loaded&&<p><small>{rows.length} saved proposals loaded{nextCursor?' · older proposals remain': ''}. Refresh returns to the latest page.</small></p>}
  {historyIssue&&<p role="alert">Older proposal history unavailable: {historyIssue} Loaded proposals are retained.</p>}
  {nextCursor&&<button disabled={busy||!!readIssue} onClick={older}>{historyIssue?'Retry older proposals':'Load older proposals'}</button>}
 </section>;
}
