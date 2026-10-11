import {researchCoverage} from './research-coverage-model.mjs';
export {sections,supported} from './stock-analysis-evidence.mjs';
export const label=s=>s.replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());
export function reportStats(run){const c=researchCoverage(run);return {stages:run?.state?.completed?.length||0,fields:c.matched+c.review+c.unavailable,supported:c.matched,unresolved:c.review,unavailable:c.unavailable};}
export function thesisDraft(run){
 if(run?.status!=='complete')throw Error('Complete the analysis first.');
 const fields=researchCoverage(run).rows.flatMap(r=>r.fields);
 const text=prefix=>fields.filter(f=>f.path.startsWith(prefix)&&f.status==='matched').map(f=>f.value).join('\n\n');
 const thesis=text('/summary/investment_thesis/');
 if(!thesis)throw Error('No source-supported investment thesis is available. Review unresolved fields first.');
 const baseline=run.baseline?.body||{};
 return {...baseline,thesis,variantView:text('/summary/variant_view/')||baseline.variantView||'',marketBaseline:text('/summary/what_market_believes/')||baseline.marketBaseline||'',changeConditions:text('/monitor/breakers/')||baseline.changeConditions||'',assumptions:baseline.assumptions||[],scenarios:baseline.scenarios||{}};
}
export function downloadText(text,name,type='text/html;charset=utf-8'){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
