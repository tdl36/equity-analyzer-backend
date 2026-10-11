// Saved lifecycle states; polling never starts or retries research.
export const agentRunPending=status=>['queued','running'].includes(status);
export const agentRunTerminal=status=>['complete','error','failed','cancelled','stopped'].includes(status);
export function agentRunHeading(status){
 return status==='queued'?'Waiting to start':status==='running'?'Saved agent progress':status==='complete'?'Run complete':agentRunTerminal(status)?'Run stopped':'Run status unavailable';
}
export function mergeAgentSnapshot(previous,runId,snapshot){
 if(!previous||previous.runId!==runId)return previous;
 if(!snapshot||typeof snapshot.status!=='string'||!Array.isArray(snapshot.newLogs)||!Number.isInteger(snapshot.totalLogs)||snapshot.totalLogs<0)return previous;
 return {...previous,readError:null,status:snapshot.status,logs:[...(previous.logs||[]),...snapshot.newLogs],decision:snapshot.decision,report:snapshot.report};
}
