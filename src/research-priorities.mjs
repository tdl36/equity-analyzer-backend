import {parseTimestamp} from './workspace-model.mjs';
export function prioritizedResearch(analyses,activities,profile,now=Date.now()) {
  const asOf=parseTimestamp(profile?.asOf),age=asOf?Math.floor((now-asOf.getTime())/86400000):null;
  const fresh=age!==null&&age>=0&&age<=30;
  const weights=new Map((profile?.positions||[]).map(p=>[p.ticker,p.weightPct]));
  const companies=new Map(analyses.map(a=>[a.ticker,a]));
  for(const tk of weights.keys())if(!companies.has(tk))companies.set(tk,{ticker:tk});
  const uniqueActivities=[...new Map(activities.map((a,i)=>[a.id||`unknown-${i}`,a])).values()];
  const rows=[...companies.values()].map(a=>{
    const date=parseTimestamp(a.updated||a.updatedAt||a.createdAt),days=date?Math.max(0,Math.floor((now-date.getTime())/86400000)):null;
    const weight=weights.get(a.ticker)??null;
    const events=uniqueActivities.filter(e=>e.ticker===a.ticker&&['pending_review','failed'].includes(e.status));
    const score=(fresh?Math.abs(weight||0)*10:0)+Math.min(days??180,180)/6+Math.min(events.length,5)*5;
    return {...a,score,weight,days,events:events.length,reasons:[...(fresh&&weight?[`${Math.abs(weight)}% ${weight<0?'short':'long'} exposure`]:[]),days===null?'No dated thesis':`${days}-day thesis`,...(events.length?[`${events.length} events awaiting attention`]:[])]};
  }).sort((a,b)=>b.score-a.score||a.ticker.localeCompare(b.ticker));
  return {rows,fresh,age};
}
