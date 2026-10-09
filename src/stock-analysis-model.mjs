export const sections=[['summary','PM summary'],['business','Business & segments'],['industry','Industry'],['financials','Financial history'],['earnings','Latest earnings'],['expectations','Consensus & revisions'],['management','Management & capital'],['valuation','Valuation'],['peers','Peers'],['scenarios','Bull / base / bear'],['debates','Market debates'],['catalysts','Catalysts'],['risks','Risks'],['monitor','Thesis monitoring'],['diligence_questions','Management questions'],['infographic','Visual summary']];
export const label=s=>s.replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());
export const supported=c=>c?.review==='supported'&&c?.passageMatched;
export function reportStats(run){const c=Object.values(run?.state?.citations||{});return {stages:run?.state?.completed?.length||0,fields:c.length,supported:c.filter(supported).length,unresolved:c.filter(v=>!supported(v)).length};}
export function thesisDraft(run){
 if(run?.status!=='complete')throw Error('Complete the analysis first.');
 const fields=run.state?.citations||{};
 const text=prefix=>Object.entries(fields).filter(([p,c])=>p.startsWith(prefix)&&supported(c)).map(([,c])=>c.statement).join('\n\n');
 const thesis=text('/summary/investment_thesis/');
 if(!thesis)throw Error('No source-supported investment thesis is available. Review unresolved fields first.');
 return {...run.baseline.body,thesis,variantView:text('/summary/variant_view/')||run.baseline.body.variantView||'',marketBaseline:text('/summary/what_market_believes/')||run.baseline.body.marketBaseline||'',changeConditions:text('/monitor/breakers/')||run.baseline.body.changeConditions||'',assumptions:run.baseline.body.assumptions||[],scenarios:run.baseline.body.scenarios||{}};
}
export function downloadText(text,name,type='text/html;charset=utf-8'){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
