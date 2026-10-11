import {sections} from './stock-analysis-model.mjs';
const unavailable=value=>value==null||String(value).trim()===''||/^(unavailable|not available|not provided|not disclosed|not recorded|n\/?a)(?:\b|[.:])/i.test(String(value));
const pointer=value=>String(value).replaceAll('~','~0').replaceAll('/','~1');
function leaves(value,path,out,depth=0) {
 if(out.length>=2000||depth>12)return;
 if(Array.isArray(value)) {if(!value.length)out.push({path,value:'',missing:true});else value.forEach((v,i)=>leaves(v,path+'/'+i,out,depth+1));}
 else if(value&&typeof value==='object') {const entries=Object.entries(value);if(!entries.length)out.push({path,value:'',missing:true});else entries.forEach(([k,v])=>leaves(v,path+'/'+pointer(k),out,depth+1));}
 else out.push({path,value:value==null?'':String(value),missing:unavailable(value)});
}
export function researchCoverage(run) {
 const report=run?.state?.report||{},citations=run?.state?.citations||{},sources=run?.sources||[],ids=new Set(sources.map(s=>s.id));
 const rows=sections.map(([id,title])=>{
  const fields=[];leaves(report[id],'/'+id,fields);
  const checked=fields.map(field=>{
   const c=citations[field.path];const evidence=(c?.evidence||[]).filter(e=>e.matched&&ids.has(e.sourceId));
   const valid=!field.missing&&c?.statement===field.value&&c.review==='supported'&&c.passageMatched&&evidence.length>0;
   const status=field.missing?'unavailable':valid?'matched':'review';
   const reason=field.missing?'No populated item in this saved section.':!c?'No field-level evidence record.':c.statement!==field.value?'The evidence record does not match the saved wording.':!evidence.length?'No matched passage linked to a frozen source.':c.reviewReason||'Check the source passage and its qualifications.';
   return {...field,status,reason,basis:c?.basis||'Unclassified',sourceIds:[...new Set(evidence.map(e=>e.sourceId))]};
  });
  const count=status=>checked.filter(f=>f.status===status).length;
  return {id,title,fields:checked,matched:count('matched'),review:count('review'),unavailable:count('unavailable')};
 });
 const used=new Set(rows.flatMap(r=>r.fields.filter(f=>f.status==='matched').flatMap(f=>f.sourceIds)));
 const sourceRows=sources.map(s=>({id:s.id,name:s.filename||'Original',url:s.sourceUrl||'',hash:s.originalHash||'',used:used.has(s.id),
  omitted:Number.isFinite(s.coverage?.omittedCharacters)?Math.max(0,s.coverage.omittedCharacters):null,
  limitation:s.coverage?.limitation||'',coverageKnown:!!s.coverage?.mode,
  extraction:(s.textExtraction?.limitations||[]).filter(x=>typeof x==='string')}));
 return {rows,sources:sourceRows,matched:rows.reduce((n,r)=>n+r.matched,0),review:rows.reduce((n,r)=>n+r.review,0),
  unavailable:rows.reduce((n,r)=>n+r.unavailable,0),noEvidence:rows.filter(r=>!r.matched).map(r=>r.title),
  limitedSources:sourceRows.filter(s=>s.omitted>0||s.limitation||s.extraction.length).length,
  unknownCoverage:sourceRows.filter(s=>!s.coverageKnown).length,baselineStale:!!run?.baselineStale};
}
const e=v=>String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
export function coverageDocument(run) {
 const c=researchCoverage(run);
 return '<!doctype html><html><meta charset="utf-8"><title>'+e(run.ticker)+' research review checklist</title><style>body{font:12pt/1.4 Calibri,sans-serif;color:#000;margin:35px}*{color:#000;font-family:Calibri,sans-serif}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#edf2f4}h2{break-after:avoid}tr{break-inside:avoid}small{font-size:9pt}a{color:#000}</style><h1>'+e(run.ticker)+' · Research review checklist</h1><p>Saved report '+e(run.id)+' · '+e(run.status)+' · '+e(run.updated_at||'Update time unavailable')+'</p><p>'+c.matched+' matched fields · '+c.review+' fields need source review · '+c.unavailable+' unavailable items. Counts include labels and periods, not just investment claims. Source matching and model review are not factual verification or investor approval.</p>'+(c.baselineStale?'<p><b>The investor case changed after this report started. Review the newer baseline.</b></p>':'')+'<table><thead><tr><th>Section</th><th>Matched</th><th>Review</th><th>Unavailable</th></tr></thead><tbody>'+c.rows.map(r=>'<tr><td>'+e(r.title)+'</td><td>'+r.matched+'</td><td>'+r.review+'</td><td>'+r.unavailable+'</td></tr>').join('')+'</tbody></table>'+c.rows.filter(r=>r.review).map(r=>'<h2>'+e(r.title)+' · Check against the originals</h2>'+r.fields.filter(f=>f.status==='review').map(f=>'<p><b>'+e(f.path)+'</b><br>'+e(f.value)+'<br><small>'+e(f.reason)+'</small></p>').join('')).join('')+'<h2>Frozen source pack</h2><p>Coverage metadata describes extraction and selection, not proof of complete document review.</p>'+c.sources.map(s=>'<p><b>'+e(s.name)+'</b><br>'+e(s.limitation||(!s.coverageKnown?'Extraction coverage metadata unavailable.':'No recorded selection omission.'))+(s.omitted!=null?'<br>Omitted characters: '+s.omitted:'')+'<br>'+s.extraction.map(e).join(' · ')+'<br><small>Original SHA-256: '+e(s.hash||'Unavailable')+'</small>'+(/^https?:\/\//i.test(s.url)?'<br><a href="'+e(s.url)+'">Observed source URL</a>':'')+'</p>').join('')+'</html>';
}
