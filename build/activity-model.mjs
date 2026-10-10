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
export function macConnected(value,now=Date.now()){const t=parseTimestamp(value);return !!t&&now-t.getTime()<180000;}
