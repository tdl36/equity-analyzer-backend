import {assignmentOutputs} from './assignment-defaults.mjs';
// Saved-output receipts only. Parent completion or collection is not output evidence.
export function labAssignmentDelivery(job) {
 const input=job?.input||{},result=job?.result||{};
 if(input.collectionOnly)return [];
 const requested=Array.isArray(input.outputs)?input.outputs:[];
 const artifacts=Array.isArray(result.artifacts)?result.artifacts:[];
 const rows=assignmentOutputs.filter(([key])=>requested.includes(key)).map(([key,label])=>{
  const saved=key==='thesis'?!!result.thesisDraftId:key==='meeting_prep'?!!result.meetingId:artifacts.some(a=>a.kind===key&&typeof a.url==='string'&&a.url);
  const error=typeof result.outputErrors?.[key]==='string'?result.outputErrors[key]:'';
  return {key,label,saved,error};
 });
 if(input.labFullNote)rows.push({key:'lab_note',label:'Full Lab note',saved:!!result.labNoteId&&result.labNoteStatus==='draft',error:typeof result.outputErrors?.lab_note==='string'?result.outputErrors.lab_note:result.labNoteStatus==='attention'?'Open the full Lab note and reconcile saved progress before resuming the assignment.':''});
 return rows.map(row=>({...row,status:row.error?(row.saved?'Saved · needs attention':'Needs attention'):row.saved?'Draft saved':job.status==='complete'?'Receipt missing':['cancelled','stopped'].includes(job.status)?'Stopped before delivery':'Not yet saved'}));
}
