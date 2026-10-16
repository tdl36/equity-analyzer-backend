import {parseTimestamp,routeHash} from './workspace-model.mjs';
const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const count=v=>Number.isSafeInteger(v)&&v>=0?v:null;
export function assignmentCollection(job,now=Date.now()){
 if(job?.input?.labSelection||job?.input?.processingProposal)return null;
 const result=job?.result||{},c=result.collection;
 if(!object(c))return null;
 const labels={queued:'Queued for browser collection',collecting:'Browser collection in progress',running:'Browser collection in progress',complete:'Browser collection reported complete',attention:'Collection needs attention',needs_auth:'AlphaSense sign-in required',cancelled:'Collection cancelled'};
 const stamp=parseTimestamp(result.macReportedAt),fresh=!!stamp&&stamp.getTime()<=now&&now-stamp.getTime()<=180000;
 const progress=object(c.sourceProgress)?c.sourceProgress:{},documents=object(progress.documents)?progress.documents:{};
 const counts=[['staged','Staged originals'],['handed_off','Handed off to iCloud'],['duplicate','Existing copies reported'],['held','Held originals']].flatMap(([key,label])=>count(documents[key])===null?[]:[{label,count:documents[key]}]);
 const tasks=Array.isArray(progress.tasks)?progress.tasks.filter(t=>object(t)&&typeof t.kind==='string'&&typeof t.status==='string').map(t=>({kind:t.kind,status:t.status,expected:count(t.expected)})):[];
 const id=result.refreshRequestId;
 return {label:labels[c.status]||'Collection state unrecognized',issue:typeof c.issue==='string'?c.issue:'',fresh,reportedAt:stamp?.toLocaleString()||'Unavailable',counts,tasks,
  verified:Array.isArray(result.sources)?result.sources.length:0,
  href:typeof id==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)?routeHash('desk',job.ticker,'collection','collection:'+id):null};
}
