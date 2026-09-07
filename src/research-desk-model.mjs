import { parseTimestamp } from './workspace-model.mjs';
export function parseTickers(text) {
  const tokens = text.toUpperCase().split(/[\s,;]+/).filter(Boolean);
  const invalid = tokens.filter(t => !/^[A-Z0-9][A-Z0-9.^=-]{0,19}$/.test(t));
  return {tickers:[...new Set(tokens)],invalid};
}
export function researchQueue(analyses, days=90, now=Date.now()) {
  return analyses.map(a=>{const date=parseTimestamp(a.updated || a.updatedAt || a.createdAt); return {...a,age:date?Math.max(0,Math.floor((now-date.getTime())/86400000)):null};})
    .filter(a=>a.age===null || a.age>=days).sort((a,b)=>(b.age??Infinity)-(a.age??Infinity));
}
export function runCounts(runs) {
  return {active:runs.filter(r=>['queued','running'].includes(r.status)).length,failed:runs.filter(r=>r.status==='error').length,complete:runs.filter(r=>r.status==='complete').length};
}
export const PLAYBOOKS = [
  {id:'refresh',name:'Coverage refresh',tag:'MULTI-COMPANY',description:'Run the research team across selected companies, then compare the conclusions with your saved theses.',steps:['Select coverage','Independent company teams','Review decisions'],view:'agents'},
  {id:'challenge',name:'Challenge the thesis',tag:'INVESTMENT COMMITTEE',description:'Revisit scenarios, assumptions and signposts against stored evidence in Investment Review.',steps:['Stored evidence','Scenario review','Decision-ready report'],view:'review'},
  {id:'meeting',name:'Management meeting',tag:'PREPARATION',description:'Organize source documents and turn the investment debate into focused management questions.',steps:['Source documents','Question generation','Meeting brief'],view:'meetingprep'},
  {id:'outputs',name:'Thesis to deliverables',tag:'CONNECTED WORKFLOW',description:'Open the existing agent workflow: update the thesis, then produce the one-pager and meeting preparation.',steps:['Thesis agent','Review gate','Parallel outputs'],view:'onepager'},
];

export function runsNeedingStatusCheck(runs,now=Date.now()) {
  return runs.filter(r=>{const created=parseTimestamp(r.createdAt);return ['queued','running'].includes(r.status)&&created&&now-created.getTime()>2*60*60*1000;});
}
