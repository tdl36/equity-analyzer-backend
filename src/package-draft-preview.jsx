import * as React from 'react';
import {claimLabel} from './onepager-edition-model.mjs';

function sections(a) {
 return [{key:'summary',title:'Company and investment case',text:a.thesis.summary},
 ...a.thesis.pillars.map((p,i)=>({key:'pillar-'+i,title:p.title,text:p.description})),
 ...a.signposts.map((p,i)=>({key:'signpost-'+i,title:'Watch: '+p.metric,text:[p.target,p.timeframe].filter(Boolean).join('\n')})),
 ...a.threats.map((p,i)=>({key:'risk-'+i,title:'Risk: '+p.threat,text:p.triggerPoints})),
 {key:'conclusion',title:'Conclusion',text:a.conclusion||''}];
}

export function PackageDraftPreview({analysis,original,claims,words}) {
 const before=new Map(sections(original).map(f=>[f.key,f]));
 const rows=sections(analysis);
 const changes=rows.filter(f=>JSON.stringify(before.get(f.key))!==JSON.stringify(f)).length+claims.filter(c=>c.proposal.statement!==words[c.correctionId]).length;
 return <section aria-label="Unsaved package preview" className="pe-preview">
 <style>{`.pe-preview{border:1px solid #bbb;padding:18px;margin:16px 0;background:#fff}.pe-preview-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.pe-preview p{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6}.pe-preview h4{font-size:20px;border-bottom:2px solid #000;padding-bottom:8px}.pe-preview h5{font-size:16px;margin:12px 0 6px}.pe-preview details{margin-top:8px}.pe-preview .pe-change{font-weight:bold;font-size:12px}@media(max-width:800px){.pe-preview-grid{grid-template-columns:1fr}.pe-preview{padding:12px}}`}</style>
 <h3>Read your draft before saving</h3>
 <p>{changes} changed sections. This is an unsaved wording preview, not an approved document or a print-layout proof. Evidence references are retained, not revalidated against your edits.</p>
 <div className="pe-preview-grid"><div><h4>Stock summary & proposed thesis</h4>{rows.map(f=>{const old=before.get(f.key),changed=JSON.stringify(old)!==JSON.stringify(f);return <article key={f.key}><h5>{f.title}</h5>{changed&&<span className="pe-change">Changed in this draft</span>}<p>{f.text||'No wording entered.'}</p>{changed&&<details><summary>Compare saved wording</summary><strong>{old.title}</strong><p>{old.text}</p></details>}</article>;})}</div>
 <div><h4>One-pager wording</h4>{claims.map(c=>{const p=c.proposal,changed=p.statement!==words[c.correctionId];return <article key={c.correctionId}><h5>{claimLabel(p.path)}</h5>{changed&&<span className="pe-change">Changed in this draft</span>}<p>{words[c.correctionId]||'No wording entered.'}</p><details><summary>Saved wording & attached evidence</summary><p>{p.statement}</p>{[p,...(p.additionalEvidence||[])].map((e,i)=><div key={i}><small>{e.locator}</small><blockquote>{e.excerpt}</blockquote></div>)}</details></article>;})}</div></div>
 </section>;
}
