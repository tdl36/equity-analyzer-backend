import {parseTimestamp} from './workspace-model.mjs';
export function activityRows(data,filter='active',query='') {
 const q=query.trim().toLowerCase();
 return (data?.items||[]).filter(r=>(filter==='all'||r.bucket===filter)&&(r.ticker+' '+r.title+' '+r.step).toLowerCase().includes(q));
}
export function activityAge(value,now=Date.now()) {
 const date=parseTimestamp(value);if(!date)return 'Time unavailable';
 const minutes=Math.max(0,Math.floor((now-date.getTime())/60000));
 return minutes<1?'Just now':minutes<60?`${minutes}m ago`:minutes<1440?`${Math.floor(minutes/60)}h ${minutes%60}m ago`:`${Math.floor(minutes/1440)}d ago`;
}
export function macConnected(value,now=Date.now()){const t=parseTimestamp(value);return !!t&&now-t.getTime()>=0&&now-t.getTime()<180000;}

export function collectionHealth(data,now=Date.now()) {
 const age=value=>{const t=parseTimestamp(value);return t?now-t.getTime():null;};
 const fresh=(value,limit)=>{const a=age(value);return a!==null&&a>=0&&a<limit;};
 const current=fresh(data?.asOf,180000),missing=data?.unavailable||[];
 const mac=current&&!missing.includes('Mac status')&&macConnected(data?.agentLastSeen,now);
 const report=current&&!missing.includes('Mac status')&&fresh(data?.collectionLastSeen,300000);
 const delivery=data?.collectionDelivery;
 const known=current&&!missing.includes('Collection delivery')&&Number.isInteger(delivery?.pending)&&delivery.pending>=0;
 const pending=known?delivery.pending:null;
 const timely=known&&(pending===0||fresh(delivery.oldestPendingAt,300000));
 return [
  {key:'mac',label:'Mac agent',ok:mac,value:mac?'Connected':'Not confirmed',detail:mac?'The agent is reporting. Browser availability and AlphaSense sign-in are checked during collection.':'Check that the Mac is awake, online and its Charlie agent is running.'},
  {key:'report',label:'Collection updates',ok:report,value:report?'Current':'Stale or unavailable',detail:report?'The Mac collection report is current. This does not mean every collection is complete.':'Check collection sync on the Mac, even if the agent is connected.'},
  {key:'delivery',label:'Requests reaching Mac',ok:timely,value:!known?'Unknown':pending===0?'No pending delivery':`${pending} awaiting receipt`,detail:!known?'Delivery could not be verified. Refresh Activity before submitting again.':pending===0?'Cloud collection commands have been acknowledged. Browser collection may still be waiting.':timely?'Waiting for the Mac to acknowledge queued collection commands.':'A collection command has not been acknowledged within five minutes, or its time is unknown. Check collection controls and Mac sync before resubmitting.'}
 ];
}
