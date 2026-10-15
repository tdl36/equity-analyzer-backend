import * as React from 'react';
const labels={'transcript':'Transcripts','broker-report':'Broker reports','press-release':'Press releases','presentation':'Presentations',unknown:'Type not recorded'};
export function ProcessingCoverage({coverage:c}){
 if(!c||c.version!==1)return <p>Evidence coverage summary unavailable for this saved proposal. Inspect its frozen sources; no thesis-incorporation assessment is available.</p>;
 const d=c.publication,r=c.thesisReferences;
 return <section aria-label="Selected evidence coverage" style={{margin:'12px 0',padding:12,border:'1px solid var(--border-color,#d8dedd)',overflowWrap:'anywhere'}}>
  <strong>Evidence coverage at proposal save</strong>
  <p>Requested collection window: {c.collectionWindow.since||'unknown'} – {c.collectionWindow.until||'unknown'}. This does not certify that every relevant source was collected.</p>
  <p>{c.total} selected originals · {c.roles.collection} from collection · {c.roles.supplemental} supplemental history · {c.roles.unknown} with role unrecorded.</p>
  <p>{d.known?`Recorded publication dates: ${d.earliest} – ${d.latest}.`:'No publication dates recorded.'} {d.unknown} of {c.total} dates unknown. This range is not continuous or exhaustive evidence coverage.</p>
  <p>{d.beforeWindow} dated before the collection window · {d.withinWindow} within it · {d.afterWindow} after it. Older sources can supply historical context; a discovery window does not limit a longitudinal thesis review.</p>
  <details><summary>Source mix and thesis references</summary>
   <ul>{Object.entries(c.kinds).filter(([,n])=>n>0).map(([kind,n])=><li key={kind}>{labels[kind]||kind}: {n}</li>)}</ul>
   <p>{r.referenced} exact-byte references found in the saved thesis register · {r.noReference} without a matching reference · {r.unknown} not checked in this saved proposal.</p>
   <p>A reference does not prove that evidence was incorporated or remains valid. No matching reference does not prove a source is new or unused. Accepted-thesis evidence coverage and freshness require investor review.</p>
  </details>
  {c.metadataRecorded<c.total&&<p>Metadata was recorded for {c.metadataRecorded} of {c.total} selected originals. Missing fields remain unknown; current library metadata has not been substituted.</p>}
 </section>;
}
