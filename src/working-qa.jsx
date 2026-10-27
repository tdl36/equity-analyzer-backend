import React,{useEffect,useState} from 'react';
import {documentHtml} from './summary-lab-format.mjs';

export function WorkingQA({api,kind,parentId,renderHtml}){
 const [row,setRow]=useState(null),[error,setError]=useState(''),[notice,setNotice]=useState(''),[busy,setBusy]=useState(false),[loaded,setLoaded]=useState(false);
 const base=`${api}/api/summary-lab`,path=`${base}/working-qa/${kind}/${encodeURIComponent(parentId)}`;
 const active=row&&['queued','running'].includes(row.status),text=row?.state?.sections?.working_qa||'',done=row?.status==='complete';
 const stalled=active&&row.updated_at&&Date.now()-Date.parse(row.updated_at)>15*60*1000;
 useEffect(()=>{let live=true,timer;setRow(null);setError('');setNotice('');setLoaded(false);
  async function load(){try{const r=await fetch(path);const d=await r.json();if(!r.ok)throw Error(d.error||'Could not load Working Notes.');if(live){setRow(d.experiment);setLoaded(true);setError('');}}catch(e){if(live)setError(e.message);}finally{if(live)timer=setTimeout(load,12000);}}
  load();return()=>{live=false;clearTimeout(timer);};
 },[path]);
 async function act(action){setBusy(true);setError('');setNotice('');try{
  if(action==='copy'){await navigator.clipboard.writeText(new DOMParser().parseFromString(documentHtml(text,renderHtml).replace(/<\/(p|li)>/g,'</$1>\n'),'text/html').body.textContent||'');setNotice('Copied.');return;}
  const url=action==='generate'?path:`${base}/${row.id}/${action==='save'?'save-to-icloud':action}`;
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(action==='save'?{section:'working_qa'}:{})});const d=await r.json();if(!r.ok)throw Error(d.error||'Request failed.');
  if(action==='generate')setRow(d.experiment);
  if(action==='save')setNotice('Word document queued for iCloud. Your Mac agent must be online to deliver it.');
  if(action==='stop')setNotice('Stop requested. Saved work is preserved.');
 }catch(e){setError(e.message+' No automatic resubmission was made.');}finally{setBusy(false);}}
 async function download(){setBusy(true);setError('');try{const r=await fetch(`${base}/${row.id}/working-qa.docx`);if(!r.ok)throw Error('Word download is unavailable.');const url=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=url;a.download='Q&A Working Notes.docx';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){setError(e.message);}finally{setBusy(false);}}
 return <details className="working-qa rounded-xl border border-white/10 overflow-hidden" style={{margin:'16px 0',padding:16}} open>
  <style>{`.working-qa button{min-height:44px;padding:8px 12px;border:1px solid #777;border-radius:6px;font:inherit;background:transparent;color:inherit;cursor:pointer}.working-qa button:disabled{opacity:.5;cursor:default}.working-qa button:focus-visible,.working-qa summary:focus-visible{outline:2px solid currentColor;outline-offset:3px}`}</style>
  <summary style={{cursor:'pointer',fontWeight:600}}>Q&amp;A · Working Notes</summary>
  <p style={{fontSize:13,opacity:.8,margin:'10px 0'}}>Short questions and compact answer bullets, alongside your complete notes.</p>
  {!row&&loaded&&<><p style={{fontSize:13}}>Generate this additional section from the saved original. Uses model credits for source review, drafting and checks; other sections stay unchanged.</p><button disabled={busy} onClick={()=>act('generate')} style={{padding:10}}>{busy?'Starting…':'Generate Q&A Working Notes'}</button></>}
  {!loaded&&!error&&<p role="status">Loading saved Working Notes…</p>}
  {active&&<p role="status">{row.state?.progress||'Queued'} <button disabled={busy} onClick={()=>act('stop')}>Stop</button></p>}
  {stalled&&<p role="alert">No progress has been saved for over 15 minutes. The run may be interrupted. No automatic restart has been requested.</p>}
  {row&&['failed','cancelled'].includes(row.status)&&<p role="alert">{row.error||'Generation stopped.'} Saved stages are preserved. Review this run in Summary Lab before resuming; an interrupted model call may have incurred credits. <a href="#view=summarylab">Open Summary Lab</a></p>}
  {done&&<div style={{display:'flex',flexWrap:'wrap',gap:12,margin:'12px 0'}}><button disabled={busy} onClick={()=>act('copy')}>Copy</button><button disabled={busy} onClick={download}>Download Word</button><button disabled={busy} onClick={()=>act('save')}>Save to iCloud</button></div>}
  {text&&<>{!done&&<p>Draft in progress; source checks may still revise it.</p>}<div className="summary-content" style={{background:'#fff',color:'#000',fontFamily:'Calibri, Carlito, sans-serif',padding:20,overflowWrap:'anywhere'}} dangerouslySetInnerHTML={{__html:documentHtml(text,renderHtml)}}/></>}
  {row&&<details style={{marginTop:12}}><summary>Private review appendix</summary><pre style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{row.state?.finalReview||'Review not yet complete.'}</pre>{(row.state?.editorialFlags?.working_qa||[]).map((f,i)=><p key={i}>{f}</p>)}</details>}
  {notice&&<p role="status">{notice}</p>}{error&&<p role="alert">{error}</p>}
 </details>;
}
