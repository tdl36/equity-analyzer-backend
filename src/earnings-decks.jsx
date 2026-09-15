import * as React from 'react';

export function EarningsDecks({api,activityId,available}) {
  const [decks,setDecks]=React.useState([]),[deck,setDeck]=React.useState(null),[body,setBody]=React.useState(null);
  const [mode,setMode]=React.useState('full'),[theme,setTheme]=React.useState('paper'),[slideIndex,setSlideIndex]=React.useState(0);
  const [busy,setBusy]=React.useState(''),[error,setError]=React.useState(''),[message,setMessage]=React.useState(''),[history,setHistory]=React.useState([]);
  const lock=React.useRef(false),attempt=React.useRef(null),alive=React.useRef(true);
  const dirty=!!deck&&JSON.stringify(body)!==JSON.stringify(deck.body);
  async function json(path,options={}) {
    const r=await fetch(`${api}${path}`,{...options,signal:AbortSignal.timeout(30000),headers:{'Content-Type':'application/json',...options.headers}});
    const data=await r.json();if(!r.ok)throw Error(data.error||'The request failed. Try again.');return data;
  }
  const select=d=>{setDeck(d);setBody(structuredClone(d.body));setSlideIndex(0);setHistory(d.history||[]);};
  const reload=async()=>{const d=await json(`/api/earnings/decks?activityId=${encodeURIComponent(activityId)}`);if(alive.current)setDecks(d.decks.sort((a,b)=>b.body.createdAt.localeCompare(a.body.createdAt)));return d;};
  React.useEffect(()=>{alive.current=true;reload().catch(e=>{if(alive.current)setError(e.message);});return()=>{alive.current=false;};},[api,activityId]);
  async function run(label,action){if(lock.current)return;lock.current=true;setBusy(label);setError('');setMessage('');try{await action();}catch(e){if(alive.current)setError(e.name==='TimeoutError'?'Response not confirmed. Reload saved decks before retrying. Your recap is unchanged.':e.message);}finally{lock.current=false;if(alive.current)setBusy('');}}
  const build=()=>run('Building deck…',async()=>{
    const config=JSON.stringify({mode,theme,activityId});if(!attempt.current||attempt.current.config!==config)attempt.current={config,id:crypto.randomUUID()};
    const d=await json('/api/earnings/decks',{method:'POST',body:JSON.stringify({activityId,mode,theme,requestId:attempt.current.id})});
    if(!alive.current)return;select(d);attempt.current=null;setMessage('Deck saved. Review the wording, then export an editable PowerPoint.');await reload();
  });
  const save=()=>run('Saving revision…',async()=>{const d=await json(`/api/earnings/decks/${deck.id}`,{method:'PUT',body:JSON.stringify({revision:deck.revision,body})});if(!alive.current)return;const position=slideIndex;select(d);setSlideIndex(position);setMessage('New revision saved. Earlier versions remain available.');await reload();});
  const open=(id,revision)=>run('Loading deck…',async()=>{const d=await json(`/api/earnings/decks/${id}${revision?`?revision=${revision}`:''}`);if(alive.current)select(d);});
  const download=()=>run('Preparing PowerPoint…',async()=>{
    const r=await fetch(`${api}/api/earnings/decks/${deck.id}/export?revision=${deck.revision}`,{signal:AbortSignal.timeout(60000)});
    if(!r.ok){const e=await r.json();throw Error(e.error||'Export failed');}const blob=await r.blob();const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`${body.ticker}_earnings_review_v${deck.revision}.pptx`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);setMessage('PowerPoint downloaded. Text and shapes are editable.');
  });
  const copy=()=>run('Copying…',async()=>{await navigator.clipboard.writeText(body.slides.map(s=>`${s.title}\n${s.items.map(t=>`• ${t}`).join('\n')}`).join('\n\n'));setMessage('All slides copied.');});
  const edit=(field,value)=>setBody(b=>({...b,slides:b.slides.map((s,i)=>i===slideIndex?{...s,[field]:value}:s)}));
  const slide=body?.slides[slideIndex];
  return <section className="earnings-decks" aria-label="Earnings presentation builder">
    <header><p className="workspace-eyebrow">RECAP → PRESENTATION</p><h3>Your earnings review, ready to present.</h3><p>Build editable slides from this saved recap. Full coverage keeps every section; the brief uses the first two paragraphs per section. Full section text stays in speaker notes.</p></header>
    <div className="ed-controls"><label>Coverage<select value={mode} onChange={e=>setMode(e.target.value)} disabled={!!busy}><option value="full">Full recap</option><option value="brief">Section brief</option></select></label><label>Slide style<select value={theme} onChange={e=>setTheme(e.target.value)} disabled={!!busy}><option value="paper">Editorial paper</option><option value="midnight">Midnight boardroom</option><option value="sage">Sage research</option></select></label><button className="workspace-primary" disabled={!available||!!busy||dirty} onClick={build}>{busy==='Building deck…'?busy:'Build earnings deck'}</button></div>
    {!available&&<p>Available when this event has a completed, saved recap.</p>}
    <p className="ed-scope">Formats existing research; does not run a new source search or approve a thesis change. The source register identifies recap inputs, not verified support for every slide claim.</p>
    <div className="ed-controls"><label>Saved decks<select aria-label="Saved earnings decks" value={deck?.id||''} disabled={!!busy||dirty} onChange={e=>e.target.value&&open(e.target.value)}><option value="">Choose a saved deck</option>{decks.map(d=><option key={d.id} value={d.id}>{new Date(d.body.createdAt).toLocaleString()} · {d.body.mode==='full'?'Full':'Brief'} · v{d.revision}</option>)}</select></label><button disabled={!!busy||dirty} onClick={()=>run('Refreshing…',async()=>{await reload();if(deck)select(await json(`/api/earnings/decks/${deck.id}`));})}>Reload saved decks</button></div>
    {error&&<p className="workspace-error" role="alert">{error}</p>}<p role="status" aria-live="polite">{busy||message}</p>
    {body&&<><div className="ed-toolbar"><strong>{body.slides.length} slides · version {deck.revision}{dirty?' · unsaved edits':''}</strong><div className="ed-controls"><button disabled={!!busy||!dirty} onClick={save}>Save new version</button><button disabled={!!busy||dirty} onClick={download}>Download PowerPoint</button><button disabled={!!busy} onClick={copy}>Copy all slides</button>{dirty&&<button disabled={!!busy} onClick={()=>setBody(structuredClone(deck.body))}>Discard edits</button>}</div></div>
      {dirty&&<p>Save or discard your edits before exporting or opening another deck.</p>}
      {body.warnings.map((w,i)=><p className="ed-scope" key={i}>{w}</p>)}
      <div className="ed-studio"><nav aria-label="Slides">{body.slides.map((s,i)=><button key={s.id} aria-current={i===slideIndex?'step':undefined} onClick={()=>setSlideIndex(i)}><small>{String(i+1).padStart(2,'0')} / {s.kind==='sources'?'Sources':s.edited?'Edited draft':'Recap draft'}</small><span>{s.title}</span></button>)}</nav><div className="ed-canvas-column">
        <article className={`ed-preview ed-${body.theme}`} aria-label={`Slide ${slideIndex+1} preview`}><p className="ed-kicker">CHARLIE / {body.ticker} / EARNINGS REVIEW</p><h4>{slide.title}</h4><ol>{slide.items.map((t,i)=><li key={i}>{t}</li>)}</ol><footer>Review draft · source register in appendix <span>{slideIndex+1} / {body.slides.length}</span></footer></article>
        {slide.kind!=='sources'&&<details className="ed-editor"><summary>Edit this slide</summary><label>Title<input value={slide.title} maxLength={180} disabled={!!busy} onChange={e=>edit('title',e.target.value)}/></label>{slide.items.map((t,i)=><label key={i}>Point {i+1}<textarea aria-label={`Point ${i+1}`} value={t} maxLength={240} rows={3} disabled={!!busy} onChange={e=>edit('items',slide.items.map((v,j)=>j===i?e.target.value:v))}/><small>{t.length} / 240 characters</small></label>)}<p>Edits remain your interpretation and are not independently verified.</p></details>}
        <details className="ed-editor"><summary>Original section and evidence scope</summary><p>{body.scope}</p><pre>{slide.notes||'No original section text recorded.'}</pre><small>Frozen recap reference: {body.recapHash.slice(0,16)}</small></details>
      </div></div>
      <div className="ed-controls"><button disabled={!!busy||dirty} onClick={()=>open(deck.id)}>Load version history</button>{history.length>0&&<label>Saved version<select value={deck.revision} disabled={!!busy||dirty} onChange={e=>open(deck.id,e.target.value)}>{history.map(h=><option key={h.revision} value={h.revision}>Version {h.revision} · {h.createdAt}</option>)}</select></label>}</div>
    </>}
  </section>;
}
