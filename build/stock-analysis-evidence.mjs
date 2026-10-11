export const sections=[['summary','PM summary'],['business','Business & segments'],['industry','Industry'],['financials','Financial history'],['earnings','Latest earnings'],['expectations','Consensus & revisions'],['management','Management & capital'],['valuation','Valuation'],['peers','Peers'],['scenarios','Bull / base / bear'],['debates','Market debates'],['catalysts','Catalysts'],['risks','Risks'],['monitor','Thesis monitoring'],['diligence_questions','Management questions'],['infographic','Visual summary']];
export const unavailable=value=>value==null||String(value).trim()===''||/^(unavailable|not available|not provided|not disclosed|not recorded|n\/?a)(?:\b|[.:])/i.test(String(value));
export const pointer=value=>String(value).replaceAll('~','~0').replaceAll('/','~1');
export function supported(c,value,sources=[]) {
 return !unavailable(value)&&c?.statement===String(value)&&c.review==='supported'&&c.passageMatched===true&&
  (c.evidence||[]).some(e=>e.matched===true&&sources.some(s=>s.id===e.sourceId));
}
