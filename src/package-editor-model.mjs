export function thesisFields(a) {
 const out=[{path:['thesis','summary'],label:'Company and investment case',value:a.thesis.summary}];
 for(const [section,rows,fields] of [['pillars',a.thesis.pillars,['title','description']],['signposts',a.signposts,['metric','target','timeframe']],['threats',a.threats,['threat','triggerPoints']]])
  rows.forEach((row,i)=>fields.forEach(field=>out.push({path:section==='pillars'?['thesis',section,i,field]:[section,i,field],label:`${section==='pillars'?'Pillar':section==='signposts'?'Signpost':'Risk'} ${i+1} · ${field==='triggerPoints'?'reconsideration trigger':field}`,value:row[field]||''})));
 out.push({path:['conclusion'],label:'Conclusion',value:a.conclusion||''});return out;
}
export function setText(analysis,path,value) {
 const next=structuredClone(analysis),allowed=thesisFields(next).some(f=>JSON.stringify(f.path)===JSON.stringify(path));
 if(!allowed||typeof value!=='string')throw Error('Choose an editable research field.');
 let at=next;for(const k of path.slice(0,-1))at=at[k];at[path.at(-1)]=value;return next;
}
export function revisionPackage(draft,analysis,requestId) {
 const p=structuredClone(draft.package);p.analysis=structuredClone(analysis);
 p.provenance={...p.provenance,editorialRevision:{parentDraftId:draft.id,requestId,kind:'investor-edited pending draft',note:'Original references retained for review; edits are not independently fact-checked.'}};
 return p;
}
export function correctionBody(claim,statement,fingerprint,requestId) {
 if(typeof statement!=='string'||!statement.trim()||statement.length>1600)throw Error('Keep each one-pager field between 1 and 1600 characters.');
 const p=claim.proposal;
 const extras=p.additionalEvidence?.map(e=>Object.fromEntries(['sourceId','excerpt','locator','originalPage'].filter(k=>e[k]!==undefined).map(k=>[k,e[k]])));
 return {...Object.fromEntries(['path','basis','sourceId','excerpt','locator','originalPage'].filter(k=>p[k]!==undefined).map(k=>[k,p[k]])), ...(extras?.length?{additionalEvidence:extras}:{}),statement,reason:'Investor edit. Retained evidence must be reviewed against the revised wording; no independent fact-check was performed.',fingerprint,requestId};
}

export function packageFields(analysis,claims,words) {
 return [...thesisFields(analysis).map(f=>({id:JSON.stringify(f.path),label:'Thesis & summary · '+f.label,text:f.value,limit:40000})),
 ...claims.map(c=>({id:c.correctionId,label:'One-pager · '+c.proposal.path,text:words[c.correctionId],limit:1600}))];
}
export function applyPackageAlternative(analysis,words,values) {
 let next=analysis;const updated={...words};
 const paths=new Map(thesisFields(analysis).map(f=>[JSON.stringify(f.path),f.path]));
 for(const [id,text] of Object.entries(values)) {
  if(paths.has(id))next=setText(next,paths.get(id),text);
  else if(Object.hasOwn(updated,id))updated[id]=text;
  else throw Error('The alternative belongs to a different package.');
 }
 return {analysis:next,words:updated};
}
