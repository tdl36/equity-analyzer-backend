const key='charlie-meeting-preferences-v1';
export function readMeetingPreferences(fallback='conference',storage=globalThis.localStorage){
  try{const v=JSON.parse(storage.getItem(key)||'{}');return {format:['conference','one_on_one','hosted_pm'].includes(v.format)?v.format:fallback,audience:['specialist','generalist'].includes(v.audience)?v.audience:'specialist'};}catch{return {format:fallback,audience:'specialist'};}
}
export function saveMeetingPreferences(format,audience,storage=globalThis.localStorage){
  try{storage.setItem(key,JSON.stringify({format,audience}));return true;}catch{return false;}
}
export function meetingJobTiming(job,now=Date.now()){
  const utc=x=>x?Date.parse(/[Zz]|[+-]\d\d:\d\d$/.test(x)?x:x+'Z'):NaN;
  const start=utc(job.createdAt),last=utc(job.updatedAt);
  const elapsed=Number.isFinite(start)?Math.max(0,Math.floor((now-start)/60000)):null;
  const quiet=Number.isFinite(last)?Math.max(0,Math.floor((now-last)/60000)):null;
  return `${elapsed===null?'':`${elapsed} min elapsed · `}${quiet===null?'':`Last checkpoint ${quiet} min ago · `}Results save to this meeting. You can leave this page.${quiet>=5?' This stage is taking longer; the last checkpoint is retained.':''}`;
}
