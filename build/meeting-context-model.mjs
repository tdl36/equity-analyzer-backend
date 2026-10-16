const record=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const id=v=>Number.isSafeInteger(v)&&v>0;
const topics=v=>Array.isArray(v)&&v.every(t=>record(t)&&typeof t.topic==='string'&&Array.isArray(t.questions)&&t.questions.every(q=>record(q)&&typeof q.question==='string'));
export function checkedMeetingContext(part,data,meetingId,questionSetId){
 if(!record(data)||data.meetingId!==meetingId)throw Error('The saved meeting identity could not be verified.');
 if(part==='answers'){
  if(!Array.isArray(data.answers)||new Set(data.answers.map(a=>a?.id)).size!==data.answers.length||data.answers.some(a=>!record(a)||!id(a.id)||a.meeting_id!==meetingId||typeof a.question!=='string'||typeof a.status!=='string'||!(a.response_notes===null||typeof a.response_notes==='string')))throw Error('Saved answer records could not be verified.');
 }else if(part==='versions'){
  if(typeof data.truncated!=='boolean'||!Array.isArray(data.versions)||data.versions.length>30||new Set(data.versions.map(v=>v?.id)).size!==data.versions.length||data.versions.some(v=>!record(v)||!id(v.id)||v.meeting_id!==meetingId||!id(v.version)||!topics(v.topics)))throw Error('Saved version history could not be verified.');
 }else if(part==='quality'){
  if(data.questionSetId!==questionSetId||['questions','highPriority','withVerifiedPassages','citedSources','availableSources','duplicateQuestions'].some(k=>!Number.isSafeInteger(data[k])||data[k]<0)||typeof data.scope!=='string')throw Error('Quality checks do not match this saved question pack.');
 }else throw Error('Unknown meeting context.');
 return data;
}
