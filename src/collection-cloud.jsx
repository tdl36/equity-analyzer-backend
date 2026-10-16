import {commandRecoveryKey,readCommand,retainCommand,clearCommand,commandReceipt,exactCommand} from './collection-command-recovery.mjs';
import {CollectionSynthesisStatus} from './collection-synthesis';
import {CollectionRequests} from './collection-requests';
import {parseTimestamp,workId} from './workspace-model.mjs';
import {CatalystWatch} from './catalyst-watch';
import {parseTickers} from './research-desk-model.mjs';
import * as React from 'react';
import {collectionHealth} from './collection-health.mjs';
const blank={ticker:'',hours:0,lookbackDays:30,kinds:['transcript','broker-report'],workflow:'thesis',topic:'',instructions:'',enabled:true};
export function CollectionCloud(props){return <CollectionCloudSession key={props.api} {...props}/>;}
function CollectionCloudSession({api,coverage=[],initialTicker='',routeWork='',onWorkChange}) {
  const recoveryKey=commandRecoveryKey(api);
  const [restored]=React.useState(()=>{try{return {body:readCommand(sessionStorage,recoveryKey),error:''};}catch(e){return {body:null,error:e.message};}});
  const [recoveryIssue,setRecoveryIssue]=React.useState(restored.error);
  const [data,setData]=React.useState(null),[cfg,setCfg]=React.useState(blank),[error,setError]=React.useState(''),[loadError,setLoadError]=React.useState(''),[message,setMessage]=React.useState(''),[busy,setBusy]=React.useState(false),[uncertain,setUncertain]=React.useState(!!restored.body||!!restored.error);
  const [bulk,setBulk]=React.useState(''),[custom,setCustom]=React.useState(false);
  const [query,setQuery]=React.useState(initialTicker),[policyOpen,setPolicyOpen]=React.useState(false);
  const policyEditor=React.useRef(null);
  const parsed=parseTickers(bulk);
  const alive=React.useRef(true),lock=React.useRef(false),pending=React.useRef(restored.body),readSerial=React.useRef(0);
  const json=async(options,query='')=>{const c=new AbortController(),timer=setTimeout(()=>c.abort(),20000);try{const r=await fetch(`${api}/api/collection/control${query}`,{...options,signal:c.signal});const d=await r.json();if(!r.ok){const e=new Error(d.error||`Request failed (${r.status})`);e.status=r.status;throw e;}return d;}finally{clearTimeout(timer);}};
  const refresh=async()=>{const serial=++readSerial.current,current=()=>alive.current&&serial===readSerial.current;try{
    const value=await json();if(!current())return;setData(value);setLoadError('');
    let body;try{body=readCommand(sessionStorage,recoveryKey);setRecoveryIssue('');}catch(e){setRecoveryIssue(e.message);setUncertain(true);return;}
    pending.current=body;setUncertain(!!body);
    if(body){
      const receipt=exactCommand(await json(undefined,'?requestId='+encodeURIComponent(body.requestId)),body);
      if(!current())return;
      if(receipt){clearCommand(sessionStorage,recoveryKey,body);pending.current=null;setUncertain(false);setMessage(`Command ${receipt.status==='applied'?'applied on Mac':receipt.status}. Its exact saved receipt was recovered; browser collection may still be pending.${typeof receipt.error==='string'&&receipt.error?' '+receipt.error:''}`);}
    }
  }catch(e){if(current())setLoadError(e.message);}};
  React.useEffect(()=>{alive.current=true;refresh();const timer=setInterval(()=>{if(!document.hidden)refresh();},15000);return()=>{alive.current=false;readSerial.current++;clearInterval(timer);};},[api]);
  const send=async(action,payload)=>{if(lock.current||recoveryIssue)return;lock.current=true;setBusy(true);setError('');
    const req=pending.current||{requestId:crypto.randomUUID(),action,payload};
    try{
      retainCommand(sessionStorage,recoveryKey,req);pending.current=req;setUncertain(true);
      const d=commandReceipt(await json({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(req)}),req);
      if(alive.current){clearCommand(sessionStorage,recoveryKey,req);pending.current=null;setUncertain(false);setMessage(`Command ${d.status}. Collection starts only after Mac processing and browser execution.`);await refresh();}
    }catch(e){if(alive.current){setError(e.message);if(e.status===400||e.status===409){try{clearCommand(sessionStorage,recoveryKey,req);pending.current=null;setUncertain(false);}catch(clearError){setRecoveryIssue(clearError.message);}}else setUncertain(true);await refresh();}}
    finally{lock.current=false;if(alive.current)setBusy(false);}
  };
  const snapshot=data?.snapshot,policies=snapshot?.policies||[];
  const health=collectionHealth(snapshot);
  const updated=parseTimestamp(data?.updatedAt);
  const stale=!updated||Date.now()-updated.getTime()<0||Date.now()-updated.getTime()>180000;
  const visiblePolicies=policies.filter(p=>!query.trim()||p.ticker.toUpperCase().includes(query.trim().toUpperCase()));
  return <section className="workspace-panel collection-controls"><p className="workspace-eyebrow">CLOUD CONTROLS / MAC COLLECTION</p><h2>Collection & coverage.</h2>
    <p className="desk-explainer">Commands are stored in Charlie and picked up by your Mac agent. AlphaSense downloads still require this Mac awake, Codex running and Chrome signed in. The scheduled browser worker picks up managed requests.</p>
    <p role="status">{updated?`Mac last reported ${updated.toLocaleString()}.`:'No Mac collection report yet.'} {stale?'No recent report; commands will wait for the Mac agent.':'Mac collection bridge is reporting.'}</p>
    <div className="workspace-panel"><h3>Automation readiness</h3><p>{health.scheduled} scheduled tickers · {health.verified}/{health.total} policies with a successful refresh recorded · {health.pending} open requests · {health.needsAttention} need attention</p>{health.overdue>0&&<p role="status">{health.overdue} queued requests have waited over 30 minutes; oldest {health.oldestMinutes} minutes. A connected Mac is not proof that browser collection is progressing. Inspect the requests below and the browser worker.</p>}<p className="desk-explainer">Counts reflect the latest Mac snapshot. A successful refresh is delivery evidence, not a guarantee of complete source coverage or research accuracy.</p></div>
    {recoveryIssue&&<p role="alert" className="workspace-error">{recoveryIssue} No new collection commands will be sent.</p>}{uncertain&&pending.current&&<p role="status">An unresolved collection command is retained in this browser tab. Check status to find its exact receipt, or retry the same command. New commands are blocked until it is resolved.</p>}
    {(error||loadError)&&<p role="alert" className="workspace-error">{error||loadError}</p>}{message&&<p role="status">{message}</p>}
    <a className="workspace-primary" href="#view=desk&section=ready" style={{display:'inline-block',marginBottom:16}}>Ready to process — choose synthesis jobs →</a>
    <CollectionSynthesisStatus api={api}/>
    <CollectionRequests focusId={workId(routeWork,'collection')} onClearFocus={()=>onWorkChange?.('')} requests={snapshot?.requests||[]} query={query} onQuery={setQuery} busy={busy||uncertain} onAction={send} available={!!snapshot}/>
    <details ref={policyEditor} open={policyOpen} onToggle={e=>setPolicyOpen(e.currentTarget.open)}><summary>Create or edit a collection policy</summary><p>Choose a reported policy below to edit its existing settings, or configure a new one. Nothing changes until you save.</p>
    <fieldset className="desk-form" disabled={busy||uncertain}><div className="desk-filter"><label>Ticker<input value={cfg.ticker} maxLength={20} onChange={e=>setCfg({...cfg,ticker:e.target.value.toUpperCase().trim()})} placeholder="MDT"/></label><label>Frequency<select value={custom?'custom':cfg.hours} onChange={e=>{setCustom(e.target.value==='custom');setCfg({...cfg,hours:e.target.value==='custom'?(cfg.hours||168):Number(e.target.value)});}}>{[[0,'Manual'],[1,'Hourly'],[4,'Every 4 hours'],[12,'Every 12 hours'],[24,'Daily'],[168,'Weekly'],[336,'Biweekly (14 days)'],[720,'Monthly (30 days)'],['custom','Custom interval']].map(([v,l])=><option key={v} value={v}>{l}</option>)}</select></label>{custom&&<label>Custom interval (hours)<input type="number" min="1" max="8760" step="1" value={cfg.hours||1} onChange={e=>setCfg({...cfg,hours:Number(e.target.value)})}/></label>}<label>Initial lookback days<input type="number" min="1" max="365" value={cfg.lookbackDays} onChange={e=>setCfg({...cfg,lookbackDays:Number(e.target.value)})}/></label></div>
    <label>Destination and workflow<select value={cfg.workflow} onChange={e=>setCfg({...cfg,workflow:e.target.value})}><option value="thesis">STOCKS → thesis proposal intake</option><option value="note">STOCKS → draft note and existing intake</option><option value="recap">CATALYSTS → event recap</option></select></label>
    {cfg.workflow==='recap'&&<label>Event folder<input value={cfg.topic} maxLength={160} placeholder="MDT F1Q27 Earnings" onChange={e=>setCfg({...cfg,topic:e.target.value})}/></label>}
    <div className="desk-filter collection-checks">{[['transcript','Event transcripts'],['broker-report','Broker reports'],['press-release','Press releases'],['presentation','Presentations']].map(([k,l])=><label key={k}><input type="checkbox" checked={cfg.kinds.includes(k)} onChange={()=>setCfg({...cfg,kinds:cfg.kinds.includes(k)?cfg.kinds.filter(v=>v!==k):[...cfg.kinds,k]})}/>{l}</label>)}<label><input type="checkbox" checked={cfg.enabled} onChange={e=>setCfg({...cfg,enabled:e.target.checked})}/>Policy enabled</label></div>
    <label>Collection instructions<textarea rows={3} maxLength={3000} value={cfg.instructions} onChange={e=>setCfg({...cfg,instructions:e.target.value})} placeholder="Prioritize earnings transcripts and material guidance changes…"/></label>
    <label><input type="checkbox" checked={!!cfg.createFolder} onChange={e=>setCfg({...cfg,createFolder:e.target.checked})}/>Create the ticker folder if it does not exist</label><button className="workspace-primary" disabled={!cfg.ticker||!cfg.kinds.length} onClick={()=>send('save',cfg)}>Save policy on Mac</button><button disabled={!cfg.ticker||!cfg.kinds.length} onClick={()=>send('save_trigger',cfg)}>Save & refresh now</button></fieldset>
    <details><summary>Configure a coverage group</summary><p className="desk-explainer">Apply the frequency, source types and workflow above to multiple tickers. Existing policies for these tickers will be replaced. The Mac validates every destination before saving the group; one invalid destination rejects the whole group.</p><fieldset disabled={busy||uncertain} className="desk-form"><label>Tickers<textarea rows={3} value={bulk} onChange={e=>setBulk(e.target.value)} placeholder="MDT, ABT, DE"/></label><button onClick={()=>setBulk([...new Set([...coverage.map(a=>a.ticker),...policies.map(p=>p.ticker)].filter(Boolean))].join(', '))}>Load covered companies</button><p>{parsed.tickers.length}/100 tickers · {cfg.hours?`Every ${cfg.hours} hours`:'Manual refresh'} · {cfg.kinds.join(', ')} · {cfg.workflow==='recap'?`CATALYSTS / ticker / ${cfg.topic}`:'STOCKS / ticker'}</p>{parsed.invalid.length>0&&<p role="alert">Invalid tickers: {parsed.invalid.join(', ')}</p>}<button className="workspace-primary" disabled={!parsed.tickers.length||parsed.tickers.length>100||!!parsed.invalid.length} onClick={()=>send('save_batch',{policies:parsed.tickers.map(ticker=>({...cfg,ticker}))})}>Save {parsed.tickers.length} ticker policies on Mac</button></fieldset></details>
    </details>
    {uncertain&&pending.current&&<button disabled={busy||!!recoveryIssue} onClick={()=>send()}>Retry same command</button>}<button onClick={refresh} disabled={busy}>Check status</button>
    <p className="desk-explainer">The existing ticker folder must exist in iCloud. A recap event subfolder may be created. Source restrictions and duplicate checks remain enforced. Research generation uses configured API credits; drafts still require review.</p>
    <details><summary>Coverage setup gaps</summary><p>Covered companies without a managed policy: {coverage.filter(a=>!policies.some(p=>p.ticker===a.ticker)).map(a=>a.ticker).join(', ')||'None'}</p><p>Mac-reported STOCKS folders: {snapshot?.folders?.stocks?.join(', ')||'Not yet reported'}</p><p>Mac-reported CATALYSTS folders: {snapshot?.folders?.catalysts?.join(', ')||'Not yet reported'}</p></details>
    <CatalystWatch api={api} policies={policies}/>
    <h3>Matching policies · {visiblePolicies.length} / {policies.length}</h3>{visiblePolicies.map(p=><div className="desk-row" key={p.ticker}><button disabled={busy||uncertain} onClick={()=>{setCfg({...p});setPolicyOpen(true);requestAnimationFrame(()=>policyEditor.current?.scrollIntoView({block:'nearest',behavior:'smooth'}));setCustom(![0,1,4,12,24,168,336,720].includes(p.hours));}}><strong>{p.ticker}</strong><span>{p.enabled?(p.hours?`Every ${p.hours} hours`:'Manual'):'Paused'} · {p.workflow==='recap'?`CATALYSTS / ${p.topic}`:'STOCKS'}</span></button><small>Last verified: {p.lastSuccess||'Never'} · Next due: {p.nextDue?new Date(p.nextDue*1000).toLocaleString():'Manual / paused'}</small><button disabled={busy||uncertain} onClick={()=>send('save',{...p,enabled:!p.enabled})}>{p.enabled?'Pause schedule':'Resume schedule'}</button><button disabled={busy||uncertain} onClick={()=>send('trigger',{ticker:p.ticker})}>Refresh now</button></div>)}
    <details><summary>Cloud commands · {data?.commands?.length||0}</summary>{(data?.commands||[]).map(c=><p key={c.id}>{c.ticker||'Coverage group'} · {c.input?.action} · {c.status==='applied'?'Applied on Mac':c.status}{c.error?` · ${c.error}`:''}</p>)}</details>

  </section>;
}
