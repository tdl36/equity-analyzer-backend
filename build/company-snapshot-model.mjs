// A projection of one saved investment-case revision. Never infer verification.
const value = v => typeof v === 'string' ? v : '';
export function companySnapshot(ticker, version) {
  const body = version?.body || {};
  const assumptions = (body.assumptions || []).map(a => ({
    ...a,
    evidence: (body.evidenceLinks || []).filter(link => link.assumptionId === a.id &&
      ['support', 'contrary', 'nextTest'].includes(link.field) && a[link.field] === link.after)
      .flatMap(link => (link.evidence || []).map(e => ({
        field: link.field, excerpt: value(e.excerpt), filename: value(e.source?.filename),
        provenance: value(link.provenance)
      })))
  }));
  return {ticker, revision: version?.revision || 0, savedAt: version?.created_at || null,
    thesis: value(body.thesis), variantView: value(body.variantView),
    marketBaseline: value(body.marketBaseline), changeConditions: value(body.changeConditions), assumptions,
    gaps: [!body.thesis && 'Investment thesis', !body.marketBaseline && 'Dated market expectations',
      !body.variantView && 'Variant view', !body.changeConditions && 'Thesis change conditions',
      !assumptions.length && 'Explicit assumptions',
      assumptions.some(a => !a.nextTest) && 'Next tests for every assumption',
      assumptions.some(a => !a.evidence.length) && 'Current excerpt links for every assumption'].filter(Boolean)};
}
export function compareSnapshots(before, after) {
  const changes = [];
  for (const [key,label] of Object.entries({thesis:'Investment thesis',variantView:'Variant view',marketBaseline:'Market expectations',changeConditions:'Change conditions'})) {
    if(before[key] !== after[key]) changes.push({label,before:before[key],after:after[key]});
  }
  const previous=new Map(before.assumptions.map(a=>[a.id,a]));
  const current=new Map(after.assumptions.map(a=>[a.id,a]));
  for(const id of new Set([...previous.keys(),...current.keys()])) {
    const old=previous.get(id), next=current.get(id);
    if(!old || !next) changes.push({label:next?'Assumption added':'Assumption removed',before:old?.claim||'',after:next?.claim||''});
    else for(const key of ['claim','evidenceType','support','contrary','nextTest','sourceReference']) {
      if(value(old[key])!==value(next[key])) changes.push({label:`${next.claim} · ${key}`,before:value(old[key]),after:value(next[key])});
    }
  }
  return changes;
}
const escape = v => String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function snapshotDocument(snapshot) {
  const paragraph=(label,text)=>`<section><h2>${escape(label)}</h2><p>${escape(text||'Not recorded')}</p></section>`;
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(snapshot.ticker)} investment map R${snapshot.revision}</title><style>body{font:16px Calibri,sans-serif;color:#000;background:#fff;max-width:1000px;margin:32px auto;padding:24px}h1{font-size:32px}h2{font-size:20px}p,blockquote{white-space:pre-wrap;overflow-wrap:anywhere}section,article{border:1px solid #ccc;padding:20px;margin:16px 0;break-inside:avoid}article{background:#f7f7f4}blockquote{border-left:3px solid #aaa;margin:16px 0;padding-left:16px}small{display:block}@media print{body{margin:0;padding:0}button{display:none}}</style></head><body><h1>${escape(snapshot.ticker)} · Investment map</h1><p>Saved case revision ${snapshot.revision} · ${escape(snapshot.savedAt||'Date unavailable')}</p><p>Investor-authored working assumptions. Excerpt links establish provenance, not truth or currentness. This is a saved-case map, not a complete company research report.</p>${paragraph('Investment thesis',snapshot.thesis)}${paragraph('Market expectations',snapshot.marketBaseline)}${paragraph('Where my view differs',snapshot.variantView)}${paragraph('What would change my mind',snapshot.changeConditions)}<h2>Assumptions → evidence → next test</h2>${snapshot.assumptions.map(a=>`<article><h2>${escape(a.claim)}</h2><small>Basis: ${escape(a.evidenceType)}</small>${paragraph('Supporting evidence',a.support)}${paragraph('Contrary evidence',a.contrary)}${paragraph('Next test',a.nextTest)}${paragraph('Manual reference · unverified',a.sourceReference)}${a.evidence.map(e=>`<blockquote><strong>${escape(e.filename||'Source name unavailable')}</strong><p>${escape(e.excerpt)}</p><small>${escape(e.provenance)}</small></blockquote>`).join('')}${!a.evidence.length?'<p>No current excerpt link recorded.</p>':''}</article>`).join('')}${paragraph('Coverage gaps',snapshot.gaps.join('; ')||'No empty tracked case fields. Full research coverage has not been assessed.')}</body></html>`;
}
