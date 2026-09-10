import {ResearchEdits} from './research-edits';
import * as React from 'react';
export function ResearchChat({api,context,onClose,onApplied}) {
  const [conversation,setConversation]=React.useState(()=>crypto.randomUUID());
  const [messages,setMessages]=React.useState([]),[history,setHistory]=React.useState([]),[analysts,setAnalysts]=React.useState([]);
  const [analyst,setAnalyst]=React.useState(''),[text,setText]=React.useState(''),[job,setJob]=React.useState(null);
  const [error,setError]=React.useState(''),[busy,setBusy]=React.useState(false),[uncertain,setUncertain]=React.useState(false);
  const panelRef=React.useRef(null);
  const alive=React.useRef(true),lock=React.useRef(false),pending=React.useRef(null),current=React.useRef(conversation);
  current.current=conversation;
  const fetchJson=async(path,options={})=>{const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);try{const r=await fetch(`${api}${path}`,{...options,signal:controller.signal});const d=await r.json();if(!r.ok){const e=new Error(d.error||`Request failed (${r.status})`);e.status=r.status;throw e;}return d;}finally{clearTimeout(timer);}};
  const list=async()=>{const d=await fetchJson(`/api/research/conversations?ticker=${encodeURIComponent(context.ticker)}&type=${encodeURIComponent(context.type)}`);if(alive.current)setHistory(d.conversations||[]);};
  const refresh=async(id=conversation)=>{try{const d=await fetchJson(`/api/research/conversations/${id}`);if(!alive.current||current.current!==id)return;
    setMessages(d.messages||[]);setJob(d.job);if(pending.current&&(d.messages||[]).some(m=>m.requestId===pending.current.requestId)){pending.current=null;setUncertain(false);setText('');}
  }catch(e){if(alive.current&&current.current===id&&e.status!==404)setError(`Could not refresh conversation: ${e.message}. Saved work may still be processing.`);}};
  React.useEffect(()=>{alive.current=true;list().catch(e=>{if(alive.current)setError(e.message);});fetchJson('/api/analysts').then(d=>{if(alive.current)setAnalysts(d.analysts||[]);}).catch(e=>{if(alive.current)setError(e.message);});return()=>{alive.current=false;};},[api,context.ticker,context.type]);
  React.useEffect(()=>{refresh();const timer=setInterval(()=>{if(!document.hidden)refresh();},4000);return()=>clearInterval(timer);},[conversation]);
  const active=job&&['queued','running'].includes(job.status);
  const send=async()=>{if(lock.current||active||(!text.trim()&&!pending.current))return;lock.current=true;setBusy(true);setError('');
    const payload=pending.current||{requestId:crypto.randomUUID(),conversationId:conversation,ticker:context.ticker,contentType:context.type,content:context.content,message:text.trim(),analystId:analyst};pending.current=payload;
    try{await fetchJson('/api/research/conversations/messages',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(alive.current&&current.current===payload.conversationId){pending.current=null;setText('');setUncertain(false);await refresh(payload.conversationId);await list();}}
    catch(e){if(alive.current){setError(e.message);if(e.status){pending.current=null;setUncertain(false);}else setUncertain(true);await refresh(payload.conversationId);}}
    finally{lock.current=false;if(alive.current)setBusy(false);}
  };
  const select=id=>{if(busy||uncertain)return;setConversation(id);setMessages([]);setJob(null);setText('');setError('');pending.current=null;};
  const stop=async()=>{if(lock.current)return;lock.current=true;setBusy(true);try{await fetchJson(`/api/research/conversations/${conversation}/stop`,{method:'POST'});await refresh();}catch(e){if(alive.current)setError(e.message);}finally{lock.current=false;if(alive.current)setBusy(false);}};
  React.useEffect(()=>{if(!onClose)return;const prior=document.activeElement;const trap=e=>{if(e.key!=='Tab')return;const items=[...panelRef.current.querySelectorAll('button:not(:disabled),select:not(:disabled),textarea:not(:disabled)')];const first=items[0],last=items[items.length-1];if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus();}};panelRef.current?.addEventListener('keydown',trap);const panel=panelRef.current;return()=>{panel?.removeEventListener('keydown',trap);prior?.focus();};},[]);
  const panel=<section ref={panelRef} className="research-chat" aria-label={`${context.ticker} analyst conversation`}>
    <header><div><p className="workspace-eyebrow">RESEARCH CONVERSATION / {context.ticker}</p><h3>Work through the investment case.</h3></div>{onClose&&<button autoFocus aria-label="Close analyst conversation" onClick={onClose}>Close ×</button>}</header>
    <p className="desk-explainer">Ask questions or request replacement wording for this {context.type}. New replies use the research shown here, the latest saved investment case and legacy thesis context, and the last 20 conversation messages. Original source files are not automatically consulted. Saved research is unchanged until you review and apply edits in its workflow.</p>
    <div className="research-chat-controls"><label>Replying analyst<select value={analyst} disabled={busy||active||uncertain} onChange={e=>setAnalyst(e.target.value)}><option value="">Research analyst</option>{analysts.map(a=><option key={a.id} value={a.id}>{a.name}</option>)}</select></label><label>Conversation<select value={history.some(h=>h.id===conversation)?conversation:''} disabled={busy||uncertain} onChange={e=>{if(e.target.value)select(e.target.value);}}><option value="">New conversation</option>{history.map(h=><option key={h.id} value={h.id}>{h.updatedAt}</option>)}</select></label><button disabled={busy||uncertain} onClick={()=>select(crypto.randomUUID())}>New conversation</button></div>
    {error&&<p role="alert" className="workspace-error">{error}</p>}
    <div className="research-chat-messages" role="log" aria-label="Analyst conversation messages">{messages.length?messages.map((m,i)=><article key={i} className={m.role==='user'?'chat-user':'chat-analyst'}><strong>{m.role==='user'?'You':m.analystName||'Research analyst'}</strong><p>{m.content}</p>{m.memoryHash&&<small className="desk-explainer">Company context at reply: {m.memoryReceipt?.length?m.memoryReceipt.map(e=>e.kind==='investment_case'?`investment case v${e.revision}`:'legacy thesis').join(' · '):'no saved case or thesis'}. Sources not reverified.</small>}</article>):<p className="workspace-empty">Start with a specific instruction: “Challenge the second thesis pillar,” or “Rewrite the conclusion to distinguish facts from estimates.”</p>}</div>
    <div role="status">{active?`Analyst ${job.status==='queued'?'queued':'working'} · you can close this panel and reopen the saved conversation.`:job?.error||''}</div>
    {active&&<><p className="desk-explainer">If processing was interrupted, stop this reply before sending another message. A provider request already running may still incur charges.</p><button disabled={busy} onClick={stop}>Stop reply delivery</button></>}
    <label className="earnings-search">Your instruction<textarea rows={3} maxLength={6000} value={text} disabled={busy||active||uncertain} onChange={e=>setText(e.target.value)} placeholder="Ask the analyst to explain, challenge or revise…"/></label>
    <ResearchEdits key={context.ticker} api={api} ticker={context.ticker} instruction={text} analystId={analyst} onApplied={onApplied}/>
    <footer><span>Uses configured model API credits. Conversation replies are proposed research, not verified source claims.</span><button onClick={()=>refresh()} disabled={busy}>Check status</button><button className="workspace-primary" disabled={busy||active||(!text.trim()&&!uncertain)} onClick={send}>{busy?'Submitting…':uncertain?'Retry same request':'Send to analyst'}</button></footer>
  </section>;
  return onClose?<div role="dialog" aria-modal="true" aria-label="Research analyst conversation" className="research-chat-backdrop" onKeyDown={e=>{if(e.key==='Escape')onClose();}}>{panel}</div>:panel;
}
