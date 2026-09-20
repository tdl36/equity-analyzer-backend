import React,{useEffect,useRef,useState} from 'react';
import {documentHtml,emailDocument,labFanoutPlan,youtubeLanguagePayload} from './summary-lab-format.mjs';

const ENGLISH_SECTIONS=[['brief','Executive Brief','brief'],['takeaways','Key Takeaways','summary'],['meeting','Meeting Summary','meeting_summary'],['questions','Follow-up Questions','questions'],['assessment','Overall Assessment','assessment']];
const KOREAN_SECTION=['korean','Korean Interpretation · 한국어 핵심 정리','korean_takeaways'];
const sectionsForMode=mode=>mode==='korean_only'?[KOREAN_SECTION]:mode==='korean_bilingual'?[...ENGLISH_SECTIONS,KOREAN_SECTION]:ENGLISH_SECTIONS;
const INTAKES=[['saved','Saved Summary'],['document','Document'],['audio','Audio'],['youtube','YouTube'],['paste','Paste text']];
const PENDING_KEY='charlie_summary_lab_pending_intake';

export function SummaryLab({api,getKey,getGeminiKey,renderHtml,pickFromICloud}){
 const [sharing,setSharing]=useState(false),[edits,setEdits]=useState({}),[sending,setSending]=useState(false);
 const [sources,setSources]=useState([]),[runs,setRuns]=useState([]),[id,setId]=useState(''),[row,setRow]=useState(null),[sid,setSid]=useState(''),[source,setSource]=useState(''),[title,setTitle]=useState(''),[focus,setFocus]=useState(''),[compare,setCompare]=useState(false),[error,setError]=useState(''),[busy,setBusy]=useState(false),[feedback,setFeedback]=useState(''),[notice,setNotice]=useState(''),[importing,setImporting]=useState(false),[importStatus,setImportStatus]=useState('');
 const [intake,setIntake]=useState('saved'),[audioFile,setAudioFile]=useState(null),[youtubeUrl,setYoutubeUrl]=useState(''),[youtubeTicker,setYoutubeTicker]=useState(''),[youtubeKorean,setYoutubeKorean]=useState(false),[youtubeKoreanOnly,setYoutubeKoreanOnly]=useState(false),[ingest,setIngest]=useState(null);
 const [expanded,setExpanded]=useState({brief:true,takeaways:true,meeting:false,questions:false,assessment:false,korean:true});
 const audioInput=useRef(null),monitoring=useRef('');
 const visibleSections=sectionsForMode(row?.state?.outputMode||'english');

 async function req(path='',body){const r=await fetch(`${api}/api/summary-lab${path}`,{...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(30000)});const d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d;}
 async function refreshLists(){const [a,b]=await Promise.all([req('/sources'),req()]);setSources(a.sources);setRuns(b.experiments);}
 useEffect(()=>{let active=true;async function load(){try{const [a,b]=await Promise.all([req('/sources'),req()]);if(active){setSources(a.sources);setRuns(b.experiments);}}catch(e){if(active)setError(e.message);}}load();const t=setInterval(load,12000);return()=>{active=false;clearInterval(t);};},[api]);
 useEffect(()=>{if(!id){setRow(null);return;}let active=true;setRow(null);async function load(){try{const d=await req('/'+id);if(active)setRow(d);}catch(e){if(active)setError(e.message);}}load();const t=setInterval(load,6000);return()=>{active=false;clearInterval(t);};},[id,api]);
 useEffect(()=>{setFeedback(row?.feedback||'');setSharing(false);setEdits({});},[row?.id]);

 // sourceJobId is set only when a completed transcription triggers this run.
 // The pending job lives in localStorage, so every tab and every reload
 // resumes it; without the reference each one starts its own paid experiment.
 async function startLab(summaryId,labTitle,outputMode='english',labFocus=focus,sourceJobId){
  const d=await req('',{summaryId:summaryId||undefined,source:summaryId?undefined:source,title:(labTitle||title).trim()||'Untitled experiment',focus:labFocus,outputMode,sourceJobId,apiKey:getKey()});
  setId(d.id);await refreshLists();return d;
 }
 async function start(){setBusy(true);setError('');setNotice('');try{await startLab(sid||undefined,title,'english',focus);}catch(e){setError(e.message);}finally{setBusy(false);}}

 async function importCloud(){
  setImporting(true);setError('');setImportStatus('Choose documents from iCloud…');
  try{
   const files=await pickFromICloud({mode:'bytes',title:'Summary Lab · choose iCloud documents'});
   if(!files?.length){setImportStatus('');return;}
   const records=[];
   for(let i=0;i<files.length;i++){
    const file=files[i];setImportStatus(`Reading ${i+1} of ${files.length}: ${file.name}`);
    if(!/\.(pdf|docx|txt|md|csv|png|jpe?g|webp)$/i.test(file.name))throw Error(`${file.name}: choose PDF, DOCX, text or image documents.`);
    const form=new FormData();form.append('files',file);form.append('apiKey',getKey()||'');
    const response=await fetch(`${api}/api/extract-summary-text`,{method:'POST',body:form,signal:AbortSignal.timeout(120000)});const data=await response.json();
    if(!response.ok||!data.text?.trim()||/^\[(Error processing|Unsupported file type|Image file:)/.test(data.text.trim()))throw Error(`${file.name}: ${data.error||'No readable text was extracted. Try a text-searchable PDF or paste the text.'}`);
    records.push(`=== SOURCE: ${file.name} ===\n${data.text}`);
   }
   setSid('');setSource(records.join('\n\n'));if(!title.trim())setTitle(files[0].name.replace(/\.[^.]+$/,'').slice(0,300));
   setImportStatus(`Imported ${files.length} document${files.length===1?'':'s'}. Ready to generate.`);
  }catch(e){setError(`${e.message} No imported documents were applied; your previous source is preserved.`);setImportStatus('Import did not finish.');}
  finally{setImporting(false);}
 }

 async function chooseAudioFromCloud(){
  try{const files=await pickFromICloud({mode:'bytes',title:'Summary Lab · choose one audio file'});const file=files?.[0];if(!file)return;if(!/\.(mp3|mp4|mpeg|mpga|m4a|wav|webm|ogg|flac)$/i.test(file.name))throw Error('Choose an MP3, M4A, WAV, MP4, MPEG, WebM, OGG or FLAC audio file.');setAudioFile(file);if(!title.trim())setTitle(file.name.replace(/\.[^.]+$/,'').slice(0,300));}catch(e){setError(e.message);}
 }

 async function monitorJob(jobId,label,jobTitle=title,jobFocus=focus,outputMode='english'){
  if(!jobId||monitoring.current===jobId)return;monitoring.current=jobId;
  try{
   for(let i=0;i<1080;i++){
    const response=await fetch(`${api}/api/transcribe-audio/${encodeURIComponent(jobId)}`,{signal:AbortSignal.timeout(20000)});const data=await response.json();
    if(!response.ok)throw Error(data.error||'Processing status is unavailable.');
    const phase=data.status||'processing';setIngest({jobId,label,phase,progress:data.progress||''});
    localStorage.setItem(PENDING_KEY,JSON.stringify({jobId,label,title:jobTitle,focus:jobFocus,outputMode}));
    if(['complete','done'].includes(phase)){
     const plan=labFanoutPlan(data);
     if(plan.error)throw Error(plan.error);
     localStorage.removeItem(PENDING_KEY);setBusy(true);
     if(plan.adoptId){
      setId(plan.adoptId);await refreshLists();
      setNotice(`${label} was transcribed and saved. Charlie already started its Summary Lab experiment for this recording, so it is shown here instead of starting a second run.`);
     }else{
      setNotice(`${label} was transcribed and saved. Improved analysis is now running.`);
      await startLab(plan.summaryId,jobTitle||label,outputMode,jobFocus,jobId);
     }
     setIngest(null);setBusy(false);return;
    }
    if(['failed','error'].includes(phase)){
     localStorage.removeItem(PENDING_KEY);setIngest(null);
     throw Error(data.error||`${label} processing failed.`);
    }
    await new Promise(resolve=>setTimeout(resolve,5000));
   }
   throw Error(`${label} is still processing after 90 minutes. Its saved job is preserved; reopen Summary Lab to resume checking.`);
  }catch(e){setError(e.message);setBusy(false);}
  finally{monitoring.current='';}
 }
 useEffect(()=>{try{const pending=JSON.parse(localStorage.getItem(PENDING_KEY)||'null');if(pending?.jobId){if(pending.title)setTitle(pending.title);if(pending.focus)setFocus(pending.focus);setIngest({...pending,phase:'checking',progress:'Reconnecting to saved job…'});monitorJob(pending.jobId,pending.label||'Source',pending.title||'',pending.focus||'',pending.outputMode||'english');}}catch{}},[api]);

 async function processAudio(){
  if(!audioFile)return;setBusy(true);setError('');setNotice('');
  try{
   const form=new FormData();form.append('file',audioFile);form.append('detailLevel','standard');form.append('apiKey',getKey()||'');form.append('geminiApiKey',getGeminiKey?.()||'');form.append('origin','summary-lab');
   const response=await fetch(`${api}/api/auto-process-audio`,{method:'POST',body:form,signal:AbortSignal.timeout(30*60*1000)});const data=await response.json();if(!response.ok)throw Error(data.error||'Audio upload failed.');
   const label=audioFile.name,jobTitle=title||label;localStorage.setItem(PENDING_KEY,JSON.stringify({jobId:data.jobId,label,title:jobTitle,focus,outputMode:'english'}));setBusy(false);await monitorJob(data.jobId,label,jobTitle,focus,'english');
  }catch(e){setError(e.message);setBusy(false);}
 }
 async function processYoutube(){
  setBusy(true);setError('');setNotice('');
  try{
   const language=youtubeLanguagePayload(youtubeKorean,youtubeKoreanOnly);
   const response=await fetch(`${api}/api/youtube-summarize`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:youtubeUrl.trim(),ticker:youtubeTicker.trim().toUpperCase(),generateKorean:language.generateKorean,koreanOnly:language.koreanOnly,apiKey:getKey()}),signal:AbortSignal.timeout(30000)});const data=await response.json();if(!response.ok)throw Error(data.error||'YouTube processing could not start.');
   const label=data.title||'YouTube video',jobTitle=title||label;if(!title.trim())setTitle(label);localStorage.setItem(PENDING_KEY,JSON.stringify({jobId:data.jobId,label,title:jobTitle,focus,outputMode:language.outputMode}));setBusy(false);await monitorJob(data.jobId,label,jobTitle,focus,language.outputMode);
  }catch(e){setError(e.message);setBusy(false);}
 }
 async function retry(){setBusy(true);setError('');try{await req('/'+id+'/retry',{apiKey:getKey()});setRow(await req('/'+id));}catch(e){setError(e.message);}finally{setBusy(false);}}
 async function stopRun(){
  setBusy(true);setError('');
  try{await req('/'+id+'/stop',{});setNotice('Stopping at the next checkpoint. Completed stages stay saved and you can resume later.');setRow(await req('/'+id));}
  catch(e){setError(e.message);}finally{setBusy(false);}
 }

 const state=row?.state||{};
 function openEmail(){setEdits({...state.sections});setSharing(true);setCompare(false);setExpanded(Object.fromEntries(visibleSections.map(([key])=>[key,true])));setNotice('Review and edit each section before sending. Edits affect this email only.');}
 async function sendEmail(){setSending(true);setError('');try{
  const creds=JSON.parse(localStorage.getItem('emailCredentials')||'{}');if(!creds.email)throw Error('Set your recipient email and Gmail credentials in Settings first.');
  const response=await fetch(`${api}/api/email-summary-section`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:creds.email,subject:`Summary Lab: ${row.title}`,title:row.title,section:'summary_lab',content:emailDocument(row.title,visibleSections.map(([k,l])=>[l,edits[k]||'']),renderHtml),smtpConfig:{use_gmail:creds.useGmail,gmail_user:creds.gmailUser,gmail_app_password:creds.gmailPassword,from_email:creds.gmailUser}})});const data=await response.json().catch(()=>({}));if(!response.ok)throw Error(data.error||'Email delivery could not be confirmed. Check your inbox before retrying.');setNotice(`${visibleSections.length===1?'The section':`All ${visibleSections.length} sections`} emailed to ${creds.email}.`);
 }catch(e){setError(e.message);}finally{setSending(false);}}
 async function copy(all=false,key='brief'){try{
  const items=all?visibleSections.map(([k,l])=>[l,(sharing?edits[k]:state.sections?.[k])||'']):[[visibleSections.find(s=>s[0]===key)?.[1]||'',(sharing?edits[key]:state.sections?.[key])||'']];
  const html=emailDocument(row.title,items,renderHtml);const doc=new DOMParser().parseFromString(html,'text/html');doc.querySelectorAll('p,h1,h2,h3,li,blockquote').forEach(el=>el.append('\n'));const plain=doc.body.textContent||'';
  if(window.ClipboardItem&&navigator.clipboard.write)await navigator.clipboard.write([new ClipboardItem({'text/html':new Blob([html],{type:'text/html'}),'text/plain':new Blob([plain],{type:'text/plain'})})]);else await navigator.clipboard.writeText(plain);setNotice('Formatted notes copied.');
 }catch{setError('Clipboard unavailable. Select the note text to copy.');}}
 function download(){const url=URL.createObjectURL(new Blob([JSON.stringify(row,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`Summary-Lab-${id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 const canGenerate=(intake==='saved'&&!!sid)||((intake==='paste'||intake==='document')&&!!source.trim());

 return <main className="summary-lab">
 <style>{`.summary-lab{box-sizing:border-box;--lab-border:rgba(153,142,119,.35);max-width:1500px;margin:0 auto;height:100%;min-height:0;overflow-y:auto;overflow-x:hidden;overscroll-behavior-y:contain;padding:32px;color:var(--text-primary,#e9e2d3);width:100%;min-width:0}.summary-lab *{box-sizing:border-box}.summary-lab h1,.summary-lab h2{font-family:Georgia,serif;line-height:1.2}.summary-lab h1{font-size:38px;margin:8px 0 14px}.summary-lab h2{font-size:24px;margin:0 0 18px}.summary-lab p{line-height:1.65}.summary-lab .muted{opacity:.72;font-size:13px}.summary-lab .eyebrow{color:#c9a857;letter-spacing:.13em;text-transform:uppercase;font-size:11px}.summary-lab .layout{display:grid;grid-template-columns:330px minmax(0,1fr);gap:24px;margin-top:28px}.summary-lab .panel{border:1px solid var(--lab-border);border-radius:14px;padding:24px;background:rgba(127,115,89,.045);min-width:0}.summary-lab label{display:block;font-size:13px;margin:16px 0 6px}.summary-lab input,.summary-lab select,.summary-lab textarea{width:100%;padding:11px;border:1px solid var(--lab-border);border-radius:7px;background:var(--bg-secondary,#211e18);color:inherit;font:inherit;min-width:0}.summary-lab .check-row{display:flex;align-items:flex-start;gap:9px;margin:14px 0 0;line-height:1.45;cursor:pointer}.summary-lab .check-row.nested{margin:9px 0 0 26px}.summary-lab .check-row input[type=checkbox]{width:17px;height:17px;min-width:17px;margin:1px 0 0;padding:0;accent-color:#c9a857}.summary-lab select option{background:#211e18;color:#eee}.summary-lab button{padding:10px 14px;min-height:44px;border:1px solid var(--lab-border);border-radius:7px;font:inherit;cursor:pointer;background:transparent;color:inherit}.summary-lab button:focus-visible,.summary-lab input:focus-visible,.summary-lab textarea:focus-visible,.summary-lab select:focus-visible{outline:2px solid #c9a857;outline-offset:3px}.summary-lab button:disabled{opacity:.45;cursor:default}.summary-lab button.primary,.summary-lab button[aria-pressed=true]{background:#c9a857;color:#18150f}.summary-lab .controls{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.summary-lab .intake-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-bottom:16px}.summary-lab .intake-grid button{padding:8px;min-height:38px;font-size:12px}.summary-lab .dropzone{padding:18px;border:1px dashed var(--lab-border);border-radius:10px;text-align:center;background:rgba(201,168,87,.035)}.summary-lab .experiment{display:block;width:100%;text-align:left;margin:10px 0;overflow-wrap:anywhere}.summary-lab .reader{font-family:Calibri,Carlito,Arial,sans-serif;font-size:11pt;line-height:1.55;overflow-wrap:anywhere;max-width:94ch;background:#fff;color:#242424;padding:28px;border:1px solid #dedbd4;border-radius:4px}.summary-lab .reader h1,.summary-lab .reader h2,.summary-lab .reader h3,.summary-lab .reader h4{font:700 11pt/1.5 Calibri,Carlito,Arial,sans-serif;margin:20px 0 8px}.summary-lab .reader p{margin:0 0 12px;line-height:1.55}.summary-lab .reader ul,.summary-lab .reader ol{padding-left:23px;margin:10px 0 16px}.summary-lab .reader li{margin:6px 0}.summary-lab .reader blockquote{border-left:3px solid #b9af94;padding-left:14px;margin:14px 0}.summary-lab .reader table{display:block;max-width:100%;overflow:auto;border-collapse:collapse}.summary-lab .reader th,.summary-lab .reader td{border:1px solid #ddd;padding:7px 9px;text-align:left}.summary-lab .email-editor{font:11pt/1.5 Calibri,Carlito,Arial,sans-serif;min-height:220px}.summary-lab .pair{display:grid;gap:24px;grid-template-columns:repeat(2,minmax(0,1fr))}.summary-lab .status{padding:14px;border-left:3px solid #c9a857;background:rgba(201,168,87,.08);margin:16px 0;overflow-wrap:anywhere}.summary-lab details.lab-section{border:1px solid var(--lab-border);border-radius:10px;margin:12px 0;padding:0;overflow:hidden}.summary-lab details.lab-section>summary{list-style:none;cursor:pointer;padding:16px 18px;display:flex;justify-content:space-between;align-items:center;gap:12px;background:rgba(127,115,89,.04)}.summary-lab details.lab-section>summary::-webkit-details-marker{display:none}.summary-lab .section-body{padding:18px}.summary-lab .chevron{display:inline-block;transition:transform .18s ease}.summary-lab details[open] .chevron{transform:rotate(90deg)}.summary-lab details.audit{border-top:1px solid var(--lab-border);padding:16px 0;margin-top:16px}.summary-lab pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:1.7}.summary-lab .original{overflow-wrap:anywhere;line-height:1.8}.summary-lab .original table{display:block;overflow:auto;max-width:100%}@media(max-width:900px){.summary-lab{padding:18px 18px 112px}.summary-lab .layout,.summary-lab .pair{grid-template-columns:1fr}.summary-lab h1{font-size:30px}.summary-lab .panel{padding:18px}.summary-lab .reader{padding:20px}.summary-lab .intake-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}`}</style>
 <div className="eyebrow">Charlie / Research experiments</div><h1>Summary Lab</h1><p>Read thoroughly. Preserve what was said. Separate what it means.</p><p className="muted">An independent trial with five familiar sections and optional Korean interpretation. Original summaries and automatic workflows remain unchanged.</p>
 {error&&<div role="alert" className="status">{error}<div><button onClick={()=>setError('')} aria-label="Dismiss error">Dismiss</button></div></div>}
 {ingest&&<div className="status" role="status"><strong>{ingest.label}</strong><div>{ingest.progress||ingest.phase}</div><p className="muted">You can leave this page. Charlie keeps the saved job and Summary Lab reconnects when you return.</p></div>}
 <div className="layout"><aside><section className="panel"><h2>Add a source</h2><div className="intake-grid" role="group" aria-label="Source type">{INTAKES.map(([key,label])=><button key={key} aria-pressed={intake===key} onClick={()=>{setIntake(key);setError('');}}>{label}</button>)}</div>
 {intake==='saved'&&<><label htmlFor="lab-source">Saved transcript or document</label><select id="lab-source" value={sid} onChange={e=>setSid(e.target.value)}><option value="">Choose a saved Summary…</option>{sources.map(s=><option key={s.id} value={s.id}>{s.title}</option>)}</select></>}
 {intake==='document'&&<><div className="dropzone"><strong>PDF, Word, text or image</strong><p className="muted">Choose one or several documents from the same connected iCloud sources.</p><button disabled={busy||importing} onClick={importCloud}>{importing?'Importing…':'Browse iCloud documents'}</button></div>{importStatus&&<p role="status" className="muted">{importStatus}</p>}{source&&<><label htmlFor="lab-document-text">Extracted source text</label><textarea id="lab-document-text" rows={6} value={source} onChange={e=>setSource(e.target.value)}/><p className="muted">{source.length.toLocaleString()} characters · no silent cutoff</p></>}</>}
 {intake==='paste'&&<><label htmlFor="lab-text">Complete source text</label><textarea id="lab-text" rows={9} value={source} onChange={e=>setSource(e.target.value)} placeholder="Paste a transcript or document text…"/><p className="muted">{source.length.toLocaleString()} characters · no silent cutoff</p></>}
 {intake==='audio'&&<><input ref={audioInput} type="file" hidden accept=".mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm,.ogg,.flac" onChange={e=>{const file=e.target.files?.[0];if(file){setAudioFile(file);if(!title.trim())setTitle(file.name.replace(/\.[^.]+$/,'').slice(0,300));}}}/><div className="dropzone"><strong>{audioFile?.name||'Audio recording'}</strong><p className="muted">MP3, M4A, WAV, MP4, MPEG, WebM, OGG or FLAC. The full transcript is saved before improved analysis begins.</p><div className="controls"><button onClick={()=>audioInput.current?.click()}>Choose file</button><button onClick={chooseAudioFromCloud}>Browse iCloud</button></div>{audioFile&&<p className="muted">{(audioFile.size/1024/1024).toFixed(1)} MB</p>}</div>{!getGeminiKey?.()&&<p className="muted">Audio transcription requires the Gemini key configured in Settings.</p>}</>}
 {intake==='youtube'&&<><label htmlFor="lab-youtube">YouTube link</label><input id="lab-youtube" type="url" value={youtubeUrl} onChange={e=>setYoutubeUrl(e.target.value)} placeholder="https://www.youtube.com/watch?v=…"/><label htmlFor="lab-youtube-ticker">Ticker · optional</label><input id="lab-youtube-ticker" value={youtubeTicker} maxLength={8} onChange={e=>setYoutubeTicker(e.target.value.toUpperCase())} placeholder="ABT"/><label className="check-row"><input type="checkbox" checked={youtubeKorean} onChange={e=>{setYoutubeKorean(e.target.checked);if(!e.target.checked)setYoutubeKoreanOnly(false);}}/><span><strong>한국어 핵심 정리도 함께 생성</strong><br/><span className="muted">English analysis plus a source-reviewed Korean interpretation.</span></span></label>{youtubeKorean&&<label className="check-row nested"><input type="checkbox" checked={youtubeKoreanOnly} onChange={e=>setYoutubeKoreanOnly(e.target.checked)}/><span><strong>한국어만 생성</strong><br/><span className="muted">Skip the five English sections and generate only the Korean interpretation.</span></span></label>}<p className="muted">Charlie retrieves the available transcript through your connected Mac, saves it, then starts the improved analysis in the selected language.</p></>}
 <label htmlFor="lab-title">Experiment name</label><input id="lab-title" value={title} maxLength={300} onChange={e=>setTitle(e.target.value)} placeholder="MMM conference · first trial"/>
 <label htmlFor="lab-focus">Optional emphasis</label><textarea id="lab-focus" rows={3} maxLength={4000} value={focus} onChange={e=>setFocus(e.target.value)} placeholder="Preserve the segment detail and management’s margin explanation."/>
 <p className="muted">Thorough review makes several passes over the complete source. Long transcripts take longer and remain saved if you leave.</p>
 {intake==='audio'?<button className="primary" disabled={busy||!audioFile||!getGeminiKey?.()} onClick={processAudio}>{busy?'Starting…':'Transcribe and analyze'}</button>:intake==='youtube'?<button className="primary" disabled={busy||!youtubeUrl.trim()} onClick={processYoutube}>{busy?'Starting…':'Fetch transcript and analyze'}</button>:<button className="primary" disabled={busy||importing||!canGenerate} onClick={start}>{busy?'Starting…':'Generate all five sections'}</button>}</section>
 <section className="panel" style={{marginTop:20}}><h2>Experiments</h2>{!runs.length&&<p className="muted">Your experiments will appear here.</p>}{runs.map(r=><button className="experiment" aria-pressed={id===r.id} key={r.id} onClick={()=>{setId(r.id);setNotice('');}}>{r.title}<div className="muted">{r.automatic?'Auto from SUMMARIES · ':''}{r.status==='complete'?'Ready to review':r.status==='cancelled'?'Stopped':r.status} · {new Date(r.created_at).toLocaleDateString()}</div></button>)}</section></aside>
 <section className="panel">{!row?<><div className="eyebrow">Independent source review</div><h2 style={{marginTop:12}}>{id?'Loading experiment…':'Your next research note starts here'}</h2><p>Add a saved Summary, document, recording, YouTube link or pasted transcript. Charlie generates Executive Brief, Key Takeaways, Meeting Summary, Follow-up Questions and Overall Assessment.</p><p className="muted">Audio and YouTube are transcribed first. Source ambiguities remain visible for review.</p></>:<>
 <div className="eyebrow">{row.version} · {row.model}</div><h2 style={{marginTop:12}}>{row.title}</h2><div className="status" role="status">{row.error||state.progress||'Queued'}<div className="muted">{Object.keys(state.parts||{}).length} / {state.totalParts||'—'} source parts reviewed · {row.status}</div></div>
 {(row.status==='failed'||row.status==='cancelled'||(row.status!=='complete'&&Date.now()-new Date(row.updated_at).getTime()>180000))&&<button disabled={busy} onClick={retry}>Resume saved experiment</button>}
 {(row.status==='queued'||row.status==='running')&&!row.cancel_requested&&<button disabled={busy} onClick={stopRun}>Stop this experiment</button>}
 {row.cancel_requested&&row.status!=='cancelled'&&<p className="muted" role="status">Stopping at the next checkpoint…</p>}
 <div className="controls"><button disabled={!Object.keys(row.baseline||{}).length} aria-pressed={compare} onClick={()=>setCompare(!compare)}>Compare with original</button><button disabled={row.status!=='complete'} onClick={()=>copy(true)}>Copy all</button><button disabled={row.status!=='complete'||sending} onClick={openEmail}>Email all sections</button><button onClick={download}>Download experiment</button></div>
 <div className="controls"><button onClick={()=>setExpanded(Object.fromEntries(visibleSections.map(([key])=>[key,true])))}>Expand all</button><button onClick={()=>setExpanded(Object.fromEntries(visibleSections.map(([key])=>[key,false])))}>Collapse all</button></div>{notice&&<p role="status">{notice}</p>}
 {sharing&&<section className="status"><strong>Email preview · {visibleSections.length===1?'Korean interpretation':`all ${visibleSections.length} sections`}</strong><p>Review any section below before sending. Edits affect this email only; audit notes and the full transcript are excluded.</p><div className="controls"><button className="primary" disabled={sending} onClick={sendEmail}>{sending?'Sending…':'Send all sections to myself'}</button><button disabled={sending} onClick={()=>{setSharing(false);setNotice('');}}>Close email preview</button></div></section>}
 {visibleSections.map(([key,label,baseline])=>{const value=(sharing?edits[key]:state.sections?.[key])||'';return <details key={key} className="lab-section" open={!!expanded[key]} onToggle={e=>{const open=e.currentTarget.open;setExpanded(current=>({...current,[key]:open}));}}><summary><span><span className="chevron" aria-hidden="true">›</span> <strong>{label}</strong></span><span className="muted">{value?'Ready':'Waiting'}</span></summary><div className="section-body"><div className="controls"><button disabled={!value} onClick={()=>copy(false,key)}>Copy section</button></div>{row.status!=='complete'&&value&&<p className="muted">Draft in progress. Source checks may still revise this section.</p>}{sharing?<><label htmlFor={`lab-edit-${key}`}>Edit for this email</label><textarea className="email-editor" id={`lab-edit-${key}`} value={value} onChange={e=>setEdits({...edits,[key]:e.target.value})}/><div className="reader" dangerouslySetInnerHTML={{__html:documentHtml(value||'This section is not available.',renderHtml)}}/></>:<div className={compare?'pair':''}>{compare&&<article><h3>Original · frozen at experiment start</h3><div className="original" dangerouslySetInnerHTML={{__html:documentHtml(row.baseline?.[baseline]||'No original section saved.',renderHtml)}}/></article>}<article>{compare&&<h3>Improved</h3>}<div className="reader" dangerouslySetInnerHTML={{__html:documentHtml(value||'This section will appear after source review and generation.',renderHtml)}}/></article></div>}</div></details>;})}
 <details className="audit"><summary>Reviewer notes and limitations</summary><pre>{state.finalReview||'Final cross-section review has not finished.'}</pre><p className="muted">Model review is not independent factual verification. Exact supporting passages are checked against saved source text.</p>{state.hierarchicalSynthesis&&<p>Long-source synthesis used consolidated records; complete part records remain below.</p>}</details>
 <details className="audit"><summary>Source record and supporting passages · {Object.keys(state.parts||{}).length} parts</summary>{Object.values(state.parts||{}).map(p=><details className="audit" key={p.id}><summary>{p.id} · characters {p.start}–{p.end}</summary><pre>{p.record}</pre><h4>Exact original passages</h4>{p.passages.map((v,i)=><blockquote key={i} className="reader">{v}</blockquote>)}<h4>Ambiguities / proposed corrections</h4><pre>{JSON.stringify(p.issues,null,2)}</pre></details>)}<details className="audit"><summary>Immutable original source</summary><pre>{row.source}</pre></details></details>
 <label htmlFor="lab-feedback">Your evaluation</label><textarea id="lab-feedback" rows={4} value={feedback} onChange={e=>setFeedback(e.target.value)} placeholder="What improved? What was lost, overstated, or harder to read?"/><button style={{marginTop:10}} onClick={async()=>{try{await req('/'+id+'/feedback',{feedback});setNotice('Evaluation saved.');}catch(e){setError(e.message);}}}>Save evaluation</button>
 </>}</section></div></main>;
}
