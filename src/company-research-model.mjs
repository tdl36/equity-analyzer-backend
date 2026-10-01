export function researchProgress(run) {
 const sections=run?.state?.sections||[];
 const claims=sections.flatMap(s=>s.claims||[]);
 return {sections:sections.length,reviewed:claims.filter(c=>c.review!=='pending').length,
   supported:claims.filter(c=>c.review==='supported'&&c.passageMatched).length,
   unresolved:claims.filter(c=>c.review!=='supported'||!c.passageMatched).length,
   stages:(run?.state?.completed||[]).length};
}
const escape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function researchDocument(run,compact=false) {
 const stats=researchProgress(run);const names=new Map((run.sources||[]).map(s=>[s.id,s.filename]));
 const sections=(run.state?.sections||[]).filter(s=>!compact||['summary','business','variant','risks','monitor','synthesis'].includes(s.id));
 return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(run.ticker)} source-backed research</title><style>body{font:16px Calibri,sans-serif;color:#000;background:#fff;max-width:1000px;margin:auto;padding:32px}article,section{border:1px solid #ccc;margin:16px 0;padding:16px;break-inside:avoid}p,blockquote{white-space:pre-wrap;overflow-wrap:anywhere}blockquote{margin:16px 0;border-left:3px solid #aaa;padding-left:16px}small{display:block}</style></head><body><h1>${escape(run.ticker)} · ${compact?'Investment map':'Research revision'}</h1><p>Research ${escape(run.id)} · ${escape(run.created_at)} · ${escape(run.status)} · Case baseline R${run.baseline?.revision||0}</p><p>Source-pack coverage only. ${stats.unresolved} claims require review. Passage matching and model review are not proof of truth. This report has not been accepted into the investment case.</p>${sections.map(s=>`<section><h2>${escape(s.title)}</h2>${s.claims.filter(c=>!compact||(c.review==='supported'&&c.passageMatched)).map(c=>`<article><strong>${escape(c.basis)} · ${c.review==='supported'&&c.passageMatched?'Passage matched + model reviewed':'NEEDS REVIEW'}</strong><p>${escape(c.statement)}</p><p>${escape(c.reviewReason||'Review pending')}</p>${c.evidence.map(e=>`<blockquote><strong>${escape(names.get(e.sourceId)||'Unknown source')} · ${e.matched?'passage matched':'UNMATCHED'}</strong><p>${escape(e.excerpt)}</p></blockquote>`).join('')}</article>`).join('')}${compact?'<p>Unresolved claims are omitted from this map; inspect the full research revision.</p>':''}<h3>Coverage gaps</h3><p>${escape(s.gaps.join('\n')||'No gaps were listed by the model; completeness is not established.')}</p></section>`).join('')}<h2>Frozen sources</h2>${(run.sources||[]).map(s=>`<p>${escape(s.filename)}<br>Original SHA-256: ${escape(s.originalHash)}<br>Observed URL: ${escape(s.sourceUrl||'Not recorded')}</p>`).join('')}</body></html>`;
}

export function researchCaseDraft(run,makeId) {
 if(run.status!=='complete'||run.baseline.revision!==0)throw Error('Initiation requires a completed research revision without an existing baseline.');
 const supported=id=>(run.state?.sections||[]).find(s=>s.id===id)?.claims.filter(c=>c.review==='supported'&&c.passageMatched)||[];
 const text=id=>supported(id).map(c=>c.statement).join('\n\n');
 return {thesis:text('summary'),variantView:text('variant'),marketBaseline:text('expectations'),changeConditions:text('monitor'),scenarios:{},
 assumptions:supported('summary').map(c=>({id:makeId(),claim:c.statement,evidenceType:'interpretation',support:'',contrary:'',nextTest:'',sourceReference:`Model-reviewed draft from research ${run.id}. Review original excerpts before classifying this claim.`}))};
}
