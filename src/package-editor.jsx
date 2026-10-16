import * as React from 'react';
import {thesisFields,setText,revisionPackage,correctionBody} from './package-editor-model.mjs';
import {claimLabel,editionDecision} from './onepager-edition-model.mjs';

export function PackageEditor({job,api}) {
 const packageRef=job.result.reviewPackage,base=`/api/research/assignments/${job.id}`;
 const key=`charlie-package-editor:${api}:${job.id}:${packageRef.requestId}`;
 const [open,setOpen]=React.useState(false),[data,setData]=React.useState(null),[analysis,setAnalysis]=React.useState(null),[words,setWords]=React.useState({}),[tab,setTab]=React.useState('thesis'),[busy,setBusy]=React.useState(false),[error,setError]=React.useState(''),[notice,setNotice]=React.useState(''),[prefs,setPrefs]=React.useState(null),[rule,setRule]=React.useState(''),[ack,setAck]=React.useState(false),[decisionAck,setDecisionAck]=React.useState(false);
 const attempt=React.useRef(null),locked=React.useRef(false),live=React.useRef(true);
 const dirty=!!data&&(JSON.stringify(analysis)!==JSON.stringify(data.draft.package.analysis)||Object.entries(words).some(([id,v])=>v!==data.edition.record.claims.find(c=>c.correctionId===id)?.proposal.statement));
 React.useEffect(()=>{live.current=true;return()=>{live.current=false;};},[]);
 React.useEffect(()=>{if(!dirty&&!attempt.current)return;const warn=e=>{e.preventDefault();e.returnValue='';};window.addEventListener('beforeunload',warn);return()=>window.removeEventListener('beforeunload',warn);},[dirty]);
 async function call(path,body) {const r=await fetch(api+path,{...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),signal:AbortSignal.timeout(60000)});const d=await r.json().catch(()=>({}));if(!r.ok)throw Error(d.error||'Unable to save. Your edits are retained; retry the same save.');return d;}
 async function act(fn){if(locked.current)return;locked.current=true;setBusy(true);setError('');try{await fn();}catch(e){if(live.current)setError(e.message);}finally{locked.current=false;if(live.current)setBusy(false);}}
 async function load(){
  const [draft,pack]=await Promise.all([call('/api/thesis-imports/'+packageRef.thesisDraftId),call(base+'/onepager-editions')]);
  const edition=pack.editions.find(e=>e.id===packageRef.editionId);
  if(!edition||draft.id!==packageRef.thesisDraftId||draft.ticker!==job.ticker||pack.assignmentId!==job.id||pack.runId!==job.result.reportId)throw Error('The saved package could not be matched. Reload the assignment.');
  if(!live.current)return;
  const nextWords=Object.fromEntries(edition.record.claims.map(c=>[c.correctionId,c.proposal.statement]));
  let stored=null;
  try {stored=JSON.parse(sessionStorage.getItem(key)||'null');}catch{throw Error('Browser draft recovery is unavailable. Enable browser storage before editing.');}
  if(stored&&(stored.fingerprint!==draft.fingerprint||stored.editionId!==edition.id))throw Error('Saved edits belong to another version. Keep this tab and reload the matching package.');
  if(stored && (JSON.stringify(thesisFields(stored.analysis).map(f=>f.path))!==JSON.stringify(thesisFields(draft.package.analysis).map(f=>f.path))||Object.keys(nextWords).some(k=>typeof stored.words?.[k]!=='string')))throw Error('Recovered edits could not be validated. Keep this tab open and copy your edits before proceeding.');
  setData({draft,pack,edition});setAnalysis(stored?.analysis||draft.package.analysis);setWords(stored?.words||nextWords);attempt.current=stored?.attempt||null;setOpen(true);
  if(stored)setNotice('Your unfinished edits were recovered. They have not been approved.');
  try {const p=await call('/api/thesis-imports/preferences');if(live.current)setPrefs(p);}catch{if(live.current)setNotice('Editing is available. Writing preferences could not be loaded.');}
 }
 function persist(a,w,saveAttempt=attempt.current){
  const value={fingerprint:data.draft.fingerprint,editionId:data.edition.id,analysis:a,words:w,attempt:saveAttempt};
  try{const encoded=JSON.stringify(value);sessionStorage.setItem(key,encoded);if(sessionStorage.getItem(key)!==encoded)throw Error();}catch{throw Error('Browser draft storage is unavailable. Keep this tab open and copy your edits before leaving.');}
 }
 function change(a,w){setAnalysis(a);setWords(w);setAck(false);try{persist(a,w);}catch(e){setError(e.message);}}
 async function save(){
  if(!ack)throw Error('Review the evidence and confirm your revised wording first.');
  if(data.draft.status!=='pending'||data.draft.stale||data.edition.decision||data.edition.stale)throw Error('This version was decided or its baseline changed. Prepare a fresh review version.');
  if(!attempt.current){const id=crypto.randomUUID();attempt.current={id,thesis:revisionPackage(data.draft,analysis,id),corrections:data.edition.record.claims.filter(c=>words[c.correctionId]!==c.proposal.statement).map(c=>({oldId:c.correctionId,body:correctionBody(c,words[c.correctionId],data.pack.fingerprint,crypto.randomUUID())})),editionId:crypto.randomUUID(),receipts:{}};}
  const a=attempt.current;persist(analysis,words,a);
  const draft=await call('/api/thesis-imports',a.thesis);a.draftId=draft.id;persist(analysis,words,a);
  for(const c of a.corrections){if(!a.receipts[c.oldId]){const r=await call(`/api/research/stock-analysis-run/${data.pack.runId}/corrections`,c.body);a.receipts[c.oldId]=r.id;persist(analysis,words,a);}}
  const eid=a.corrections.length?a.editionId:data.edition.id;
  if(a.corrections.length){await call(base+'/onepager-editions',{...Object.fromEntries(['assignmentId','runId','fingerprint','fundamentalsDigest','parentId'].map(k=>[k,data.pack[k]])),requestId:eid,correctionIds:data.edition.record.correctionIds.map(id=>a.receipts[id]||id)});}
  // Validate the compact rendering before making it the visible package.
  await call(base+`/onepager-editions/${eid}/view?layout=recovery`);
  await call(base+'/review-package',{requestId:a.id,previousRequestId:packageRef.requestId,thesisDraftId:a.draftId,editionId:eid});
  sessionStorage.removeItem(key);attempt.current=null;setNotice('New package version saved. Reloading the matching previews…');window.location.reload();
 }
 function discard(){if(attempt.current||!window.confirm('Discard the unsaved wording edits in this tab? Saved versions will remain unchanged.'))return;sessionStorage.removeItem(key);setAnalysis(data.draft.package.analysis);setWords(Object.fromEntries(data.edition.record.claims.map(c=>[c.correctionId,c.proposal.statement])));setAck(false);setNotice('Unsaved edits discarded. The saved package is unchanged.');}
 async function preference(rules){const value=await call('/api/thesis-imports/preferences',{rules,fingerprint:prefs.fingerprint});if(live.current){setPrefs(value);setRule('');setNotice('Writing preferences saved for future stock research and thesis preparation. Existing drafts are unchanged.');}}
 async function decide(action){const body=editionDecision(data.edition,data.pack,action,crypto.randomUUID());await call(base+`/onepager-editions/${data.edition.id}/decision`,body);setDecisionAck(false);setNotice('One-pager decision saved. Thesis approval is separate.');await load();}
 return <section className="package-editor"><style>{`.package-editor{padding:20px 24px;border-bottom:1px solid #ccc;font-family:Calibri,sans-serif;color:#000}.package-editor *{box-sizing:border-box;color:#000}.package-editor h3{margin:0 0 8px}.package-editor textarea{display:block;width:100%;font:16px/1.5 Calibri,sans-serif;padding:12px;border:1px solid #aaa;border-radius:4px;background:#fff;resize:vertical}.package-editor label{display:block;margin:14px 0}.package-editor fieldset{border:0;padding:0;min-width:0}.package-editor article{padding:14px 0;border-top:1px solid #ddd}.package-editor .pe-actions{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.package-editor button{min-height:42px}.package-editor .pe-note{padding:12px;background:#f2f1eb;line-height:1.5}.package-editor summary{cursor:pointer}.package-editor blockquote{white-space:pre-wrap;overflow-wrap:anywhere}.package-editor a{color:#000!important;text-decoration:underline}@media(max-width:650px){.package-editor{padding:16px}.package-editor .pe-actions>button{flex:1}}`}</style>
 {dirty&&<p role="status">Unsaved wording changes · saved locally in this browser tab.</p>}
 <h3>Your research, your wording</h3><p>Edit the draft here, save a new version, then review before approving. Earlier versions stay preserved.</p>
 {!open?<button disabled={busy} onClick={()=>act(load)}>{busy?'Opening editor…':'Edit & review this package'}</button>:<>
 <div className="pe-actions"><button disabled={busy} onClick={()=>setTab('thesis')}>Stock summary & thesis</button><button disabled={busy} onClick={()=>setTab('brief')}>One-pager</button><button disabled={busy} onClick={()=>setTab('preferences')}>Writing preferences</button><button disabled={busy} onClick={()=>setOpen(false)}>Close editor · keep edits</button></div>
 {tab==='preferences'?<><h4>What should Charlie remember?</h4><p>Save reusable writing rules explicitly. Stock-specific forecasts and investment judgments stay in the draft. These rules guide future stock research and thesis preparation; this is preference memory, not model training.</p>{prefs?<><ul>{prefs.rules.map((r,i)=><li key={i}>{r} <button disabled={busy} onClick={()=>act(()=>preference(prefs.rules.filter((_,n)=>n!==i)))}>Remove</button></li>)}</ul><label>Remember for future writing<textarea maxLength={500} rows={3} value={rule} onChange={e=>setRule(e.target.value)} placeholder="For example: keep broker names in source references, never in the narrative."/></label><button disabled={busy||!rule.trim()||prefs.rules.length>=20} onClick={()=>act(()=>preference([...prefs.rules,rule.trim()]))}>Remember this preference</button></>:<button disabled={busy} onClick={()=>act(async()=>setPrefs(await call('/api/thesis-imports/preferences')))}>Load writing preferences</button>}</>:<>
 {(data.draft.status!=='pending'||data.draft.stale||data.edition.decision||data.edition.stale)&&<p role="status">This saved version has been decided or its baseline has changed. Editing is disabled to preserve that decision; a fresh research revision is required.</p>}
 <p className="pe-note">{tab==='thesis'?'The stock summary and proposed thesis share these sections. Saving updates both. Review the one-pager separately: concise wording is never silently rewritten.':'Edit the one-pager’s saved claims. Evidence references remain attached; changing wording does not prove the new claim. Keep each field concise.'}</p>
 <fieldset disabled={busy||!!attempt.current||data?.draft.status!=='pending'||data?.draft.stale||data?.edition.decision||data?.edition.stale}>
 {tab==='thesis'?thesisFields(analysis).map(f=><label key={JSON.stringify(f.path)}>{f.label}<textarea rows={f.path.at(-1)==='summary'?8:3} maxLength={40000} value={f.value} onChange={e=>change(setText(analysis,f.path,e.target.value),words)}/></label>):data.edition.record.claims.map(c=><article key={c.correctionId}><label>{claimLabel(c.proposal.path)}<textarea rows={3} maxLength={300} value={words[c.correctionId]} onChange={e=>change(analysis,{...words,[c.correctionId]:e.target.value})}/></label><small>{words[c.correctionId].length}/300 characters</small><details><summary>Original wording & evidence</summary><p>{c.proposal.statement}</p>{[c.proposal,...(c.proposal.additionalEvidence||[])].map((e,i)=><div key={i}><p>{e.locator}</p><blockquote>{e.excerpt}</blockquote></div>)}</details></article>)}
 </fieldset>
 <label><input type="checkbox" checked={ack} disabled={busy} onChange={e=>setAck(e.target.checked)}/> I checked my revised wording against its evidence and reviewed both the summary/thesis and the one-pager.</label>
 <div className="pe-actions"><button disabled={busy||!ack||(!dirty&&!attempt.current)} onClick={()=>act(save)}>{attempt.current?'Retry saving this exact revision':'Save new package version'}</button>{dirty&&!attempt.current&&<button disabled={busy} onClick={discard}>Discard unsaved edits</button>}</div>
 <details><summary>Thesis evidence and source limitations</summary><p>{(data.draft.package.provenance?.limitations||[]).join(' ')}</p>{data.draft.package.sourceRegister.map(s=><p key={s.id}>{s.id} · {s.filename||s.name} · pages {(s.reviewedPdfPages||[]).join(', ')}</p>)}</details>
 <p>Saving does not approve research. Unfinished edits are retained in this browser tab; save a package version to retain them across devices.</p>
 {!dirty&&!attempt.current&&<details><summary>Decide on this saved version</summary><p>Thesis approval changes the saved detailed thesis. One-pager acceptance records a separate editorial decision.</p><a href={`?thesisDraft=${packageRef.thesisDraftId}#view=thesisimports&ticker=${job.ticker}`}>Compare and approve or dismiss the thesis</a>{data.pack.canDecide&&!data.edition.decision&&!data.edition.stale&&<><label><input type="checkbox" checked={decisionAck} onChange={e=>setDecisionAck(e.target.checked)}/> I reviewed this exact one-pager’s wording, evidence and scenario assumptions.</label><button disabled={busy||!decisionAck} onClick={()=>act(()=>decide('accept'))}>Accept this one-pager</button><button disabled={busy||!decisionAck} onClick={()=>act(()=>decide('reject'))}>Reject this one-pager</button></>}{!data.pack.canDecide&&<p>Sign in as the investor to record a one-pager decision.</p>}{data.edition.decision&&<p>One-pager decision: {data.edition.decision.action}.</p>}</details>}
 </>}
 </>}{error&&<p role="alert">{error}</p>}{notice&&<p role="status">{notice}</p>}
 </section>;
}
