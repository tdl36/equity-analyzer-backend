import * as React from 'react';
export function ResearchRestore({api,record,onClose,onRestored}) {
  const [preview,setPreview]=React.useState(null),[error,setError]=React.useState(''),[busy,setBusy]=React.useState(false),[result,setResult]=React.useState(null),[uncertain,setUncertain]=React.useState(false);
  const pending=React.useRef(null),locked=React.useRef(false);
  const panel=React.useRef(null);
  React.useEffect(()=>{panel.current?.scrollIntoView({behavior:'smooth',block:'start'});panel.current?.focus({preventScroll:true});},[record.kind,record.recordId]);
  const json=async(path,options={})=>{const r=await fetch(`${api}${path}`,{...options,signal:AbortSignal.timeout(30000)});const d=await r.json();if(!r.ok){const e=Error(d.error||`Request failed (${r.status})`);e.status=r.status;throw e;}return d;};
  React.useEffect(()=>{let alive=true;setPreview(null);setError('');setResult(null);pending.current=null;setUncertain(false);
    json(`/api/research/restoration-preview/${record.kind}/${encodeURIComponent(record.recordId)}`).then(d=>{if(alive)setPreview(d);}).catch(e=>{if(alive)setError(e.message);});return()=>{alive=false;};
  },[api,record.kind,record.recordId]);
  const restore=async()=>{if(locked.current||!preview)return;locked.current=true;setBusy(true);setError('');const d=preview;
    const body=pending.current||{requestId:crypto.randomUUID(),kind:d.kind,sourceId:d.source.id,sourceHash:d.source.hash,latestId:d.latest.id,latestHash:d.latest.hash};pending.current=body;
    try{const saved=await json('/api/research/restorations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});setResult(saved);pending.current=null;setUncertain(false);onRestored?.();}
    catch(e){setError(e.message);if(e.status){pending.current=null;setUncertain(false);}else setUncertain(true);}
    finally{locked.current=false;setBusy(false);}
  };
  return <section ref={panel} tabIndex={-1} className="workspace-panel research-restoration" aria-label="Restore a historical research version"><div className="workspace-section-heading"><div><p className="workspace-eyebrow">VERSION RESTORATION</p><h3>Review before restoring.</h3></div><button disabled={busy||uncertain} onClick={onClose}>Close preview</button></div>
    <p>Restoration creates a new saved version and preserves every original. Notes return as drafts. Investment reviews retain historical assumptions and require a fresh quality review; restoration does not refresh prices, evidence or estimates.</p>
    {error&&<p role="alert" className="workspace-error">{error}</p>}{!preview&&!error&&<p role="status">Loading the saved versions…</p>}
    {preview&&<><div className="evidence-compare"><div><h4>Selected historical version</h4><p>{preview.source.createdAt} · {preview.source.status}</p><pre>{preview.source.markdown||'No saved narrative preview.'}</pre>{preview.source.previewTruncated&&<p>Preview limited to the first 120,000 characters; restoration uses the complete saved version.</p>}</div><div><h4>Latest saved version</h4><p>{preview.latest.createdAt} · {preview.latest.status}</p><pre>{preview.latest.markdown||'No saved narrative preview.'}</pre>{preview.latest.previewTruncated&&<p>Preview limited to the first 120,000 characters.</p>}</div></div>
    {result?<p role="status">{result.kind==='note'?'A restored draft is ready in Research Pipeline.':'A historical review version has been saved with a fresh-review requirement.'} The original versions remain available.</p>:<button className="workspace-primary" disabled={busy||preview.source.id===preview.latest.id} onClick={restore}>{busy?'Restoring…':uncertain?'Retry same restoration request':preview.source.id===preview.latest.id?'Already the latest saved version':preview.kind==='note'?'Restore as a new note draft':'Restore as a historical review version'}</button>}</>}
  </section>;
}
