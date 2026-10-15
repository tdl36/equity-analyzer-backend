import {validWork,routeHash} from './workspace-model.mjs';
const record=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const statuses=['queued','running','awaiting_approval','applied','dismissed','reverted','failed'];
const identity=(j,ticker)=>record(j)&&j.ticker===ticker&&!!validWork('desk','evidence','amendment:'+j.id);
export function linkedProposals(data,ticker,workId){
 if(!record(data)||data.workId!==workId||typeof data.truncated!=='boolean'||!Array.isArray(data.jobs)||data.jobs.length>30||new Set(data.jobs.map(j=>j?.id)).size!==data.jobs.length||data.jobs.some(j=>!identity(j,ticker)||!statuses.includes(j.status)||!record(j.work_evidence)||j.work_evidence.id!==workId||!Number.isSafeInteger(j.work_evidence.revision)||j.work_evidence.revision<1)||!('activeProposal' in data)||(data.activeProposal!==null&&(!identity(data.activeProposal,ticker)||!['queued','running','awaiting_approval'].includes(data.activeProposal.status))))throw Error('Linked proposal status could not be verified. Previously loaded links remain available; preparing research is paused until the read succeeds.');
 return data;
}
export const proposalLink=(ticker,id)=>routeHash('desk',ticker,'evidence',validWork('desk','evidence','amendment:'+id));
