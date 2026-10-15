import {validWork,routeHash} from './workspace-model.mjs';
const record=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const statuses=['queued','running','awaiting_approval','applied','dismissed','reverted','failed'];
const identity=(j,ticker)=>record(j)&&j.ticker===ticker&&!!validWork('desk','evidence','amendment:'+j.id);
export function linkedProposals(data,ticker,workId){
 if(!record(data)||data.workId!==workId||typeof data.truncated!=='boolean'||!Array.isArray(data.jobs)||data.jobs.length>30||new Set(data.jobs.map(j=>j?.id)).size!==data.jobs.length||data.jobs.some(j=>!identity(j,ticker)||!statuses.includes(j.status)||!record(j.work_evidence)||j.work_evidence.id!==workId||!Number.isSafeInteger(j.work_evidence.revision)||j.work_evidence.revision<1)||!('activeProposal' in data)||(data.activeProposal!==null&&(!identity(data.activeProposal,ticker)||!['queued','running','awaiting_approval'].includes(data.activeProposal.status))))throw Error('Linked proposal status could not be verified. Previously loaded links remain available; preparing research is paused until the read succeeds.');
 return data;
}
export const proposalLink=(ticker,id)=>routeHash('desk',ticker,'evidence',validWork('desk','evidence','amendment:'+id));

// Store only submission identity and the saved work reference, never answer bodies or credentials.
export const proposalIntentKey=(api,ticker,workId)=>'charlie-meeting-proposal-v1:'+JSON.stringify([api,ticker,workId]);
const validIntent=(p,workId)=>record(p)&&Object.keys(p).sort().join(',')==='instructions,requestId,workEvidence'&&
 typeof p.requestId==='string'&&!!validWork('desk','evidence','amendment:'+p.requestId)&&p.instructions===''&&
 record(p.workEvidence)&&Object.keys(p.workEvidence).sort().join(',')==='id,revision'&&p.workEvidence.id===workId&&
 Number.isSafeInteger(p.workEvidence.revision)&&p.workEvidence.revision>0;
export function readProposalIntent(storage,key,workId){
 const raw=storage.getItem(key);if(raw===null)return null;
 let p;try{p=JSON.parse(raw);}catch{throw Error('Saved proposal request is unreadable. Submission is paused; do not clear browser storage while its outcome is unresolved.');}
 if(!validIntent(p,workId))throw Error('Saved proposal request does not match this follow-up. Submission is paused.');
 return p;
}
export function reserveProposalIntent(storage,key,workId,payload){
 const existing=readProposalIntent(storage,key,workId);if(existing)return existing;
 if(!validIntent(payload,workId))throw Error('Proposal request could not be verified.');
 const raw=JSON.stringify(payload);storage.setItem(key,raw);
 if(storage.getItem(key)!==raw)throw Error('Proposal request could not be saved safely. No submission was sent.');
 return payload;
}
export function proposalReceipt(data,ticker,payload){
 if(!record(data)||!Array.isArray(data.jobs)||data.jobs.length>1)throw Error('Exact proposal receipt could not be verified.');
 if(!data.jobs.length)return null;
 const j=data.jobs[0];
 if(!identity(j,ticker)||j.id!==payload.requestId||!statuses.includes(j.status)||!record(j.work_evidence)||
 j.work_evidence.id!==payload.workEvidence.id||j.work_evidence.revision!==payload.workEvidence.revision)
 throw Error('Exact proposal receipt does not match the saved request. Submission remains paused.');
 return j;
}
export function clearProposalIntent(storage,key,workId,payload){
 const current=readProposalIntent(storage,key,workId);
 if(!current||JSON.stringify(current)!==JSON.stringify(payload))throw Error('Saved proposal request changed during reconciliation. Check status again.');
 storage.removeItem(key);if(storage.getItem(key)!==null)throw Error('Saved proposal receipt was found, but local reconciliation could not be saved.');
}
