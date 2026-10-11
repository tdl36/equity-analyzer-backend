// Saved lifecycle states; polling never starts or retries research.
export const agentRunPending=status=>['restoring','queued','running'].includes(status);
export const agentRunTerminal=status=>['complete','error','failed','cancelled','stopped'].includes(status);
export function agentRunHeading(status){
 return status==='restoring'?'Restoring saved run':status==='queued'?'Waiting to start':status==='running'?'Saved agent progress':status==='complete'?'Run complete':agentRunTerminal(status)?'Run stopped':'Run status unavailable';
}
export function mergeAgentSnapshot(previous,runId,snapshot){
 if(!previous||previous.runId!==runId)return previous;
 if(!snapshot||typeof snapshot.status!=='string'||!Array.isArray(snapshot.newLogs)||!Number.isInteger(snapshot.totalLogs)||snapshot.totalLogs<0)return previous;
 return {...previous,readError:null,status:snapshot.status,logs:[...(previous.logs||[]),...snapshot.newLogs],decision:snapshot.decision,report:snapshot.report};
}

export function savedSubmission(storage,payload,newId=()=>crypto.randomUUID()){
 const key='charlie-analyst-submission:'+JSON.stringify(payload);
 let id=storage.getItem(key);
 if(!id){id=newId();storage.setItem(key,id);}
 if(!/^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(id))throw new Error('Saved submission identity is invalid. Inspect Activity before submitting again.');
 return {key,id};
}

const SELECTION='charlie-analyst-selected-run';
export function selectedAgentRun(storage){
 try{
  const saved=JSON.parse(storage.getItem(SELECTION)||'null');
  if(!saved||!/^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(saved.runId)||typeof saved.ticker!=='string')return null;
  return {runId:saved.runId,ticker:saved.ticker,status:'restoring',logs:[],decision:null};
 }catch{return null;}
}
export function rememberAgentRun(storage,runId,ticker){storage.setItem(SELECTION,JSON.stringify({runId,ticker}));}

export function observeAgentRun({api,runId,fetcher=fetch,onSnapshot,onError,onComplete,interval=setInterval,cancel=clearInterval}){
 let active=true,reading=false,cursor=0,first=true,controller;
 const read=async()=>{
  if(!active||reading)return;
  reading=true;controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),15000);
  try{
   const response=await fetcher(`${api}/api/agents/status/${encodeURIComponent(runId)}?since=${cursor}`,{signal:controller.signal});
   if(!response.ok)throw Error('Saved run unavailable');
   const snapshot=await response.json();
   if(!Array.isArray(snapshot.newLogs)||!Number.isInteger(snapshot.totalLogs)||snapshot.totalLogs<cursor||typeof snapshot.status!=='string')throw Error('Invalid saved run state');
   if(!active)return;
   onSnapshot(snapshot,first);first=false;cursor=snapshot.totalLogs;
   if(agentRunTerminal(snapshot.status)){active=false;cancel(timer);onComplete();}
  }catch{if(active)onError('Status refresh failed. Showing the last saved state; Charlie will retry this read.');}
  finally{clearTimeout(timeout);reading=false;}
 };
 const timer=interval(read,3000);read();
 return ()=>{active=false;cancel(timer);controller?.abort();};
}
