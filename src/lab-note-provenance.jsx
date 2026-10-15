import * as React from 'react';
export default function NoteProvenance({value}) {
 if(!value)return <p>This note has no structured frozen research-purpose or source-date record. Its scope is not inferred from its title.</p>;
 const event=value.purpose.kind==='event_note'?value.purpose.event:null;
 return <section aria-label="Saved research scope" style={{background:'#fff',color:'#000',fontFamily:'Calibri, sans-serif',padding:16,margin:'16px 0',border:'1px solid #ddd',overflowWrap:'anywhere'}}>
  <h4 style={{marginTop:0}}>{event?'Event note / debrief':'Longitudinal thesis review'}</h4>
  {event?<><p><strong>{event.title}</strong> · {event.date}</p><p>Event-focused analysis with historical context. This note does not update the thesis or establish thesis freshness. Thesis implications require a separate proposal.</p></>:<p>Discovery dates limit new-source collection, not analytical history. This note does not apply thesis changes.</p>}
  <details><summary>Frozen source context · {value.sourceEvidence?.sources.length??'not recorded'}</summary>
   <p>Publication dates are not financial periods. Thesis-register references do not prove incorporation or current validity. Missing references do not establish that evidence is new.</p>
   {value.sourceEvidence?<ul style={{paddingLeft:20}}>{value.sourceEvidence.sources.map(s=><li key={s.id} style={{marginBottom:12}}><strong>{s.filename}</strong><br/>{s.role==='collection'?'Collected evidence':'Supplemental history'} · Publication: {s.publishedDate||'Unknown'}<br/>Publisher: {s.publisher||'Unknown'} · Type: {s.sourceKind||'Unknown'}<br/>{s.referencedBySavedThesis===true?'Referenced by saved thesis register':s.referencedBySavedThesis===false?'No saved thesis-register reference':'Thesis-register reference unknown'}</li>)}</ul>:<p>Source dates and roles were not recorded for this note.</p>}
  </details>
 </section>;
}
