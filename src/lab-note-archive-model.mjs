import {validWork} from './workspace-model.mjs';
export function noteArchivePage(value,ticker) {
 if(value?.readOnly!==true||value.ticker!==ticker||!Array.isArray(value.jobs)||value.jobs.length>30||
    !(value.nextCursor===null||typeof value.nextCursor==='string'&&value.nextCursor.length>0&&value.nextCursor.length<=512))throw Error('Saved note archive is incomplete. Retry this read.');
 const seen=new Set();
 for(const row of value.jobs){
  if(!row||row.ticker!==ticker||!validWork('pipelinelab','','labnote:'+row.noteId)||seen.has(row.noteId)||
     !['draft','attention','queued','running'].includes(row.status)||typeof row.title!=='string'||typeof row.createdAt!=='string'||!Number.isFinite(Date.parse(row.createdAt))||
     ![row.revision,row.acceptedRevision].every(v=>v===null||Number.isSafeInteger(v)&&v>0))throw Error('Saved notes do not match this company archive. No other notes have been substituted.');
  seen.add(row.noteId);
 }
 return {rows:value.jobs,cursor:value.nextCursor};
}
export function appendNoteArchive(before,page) {
 const seen=new Set(before.map(row=>row.noteId));return [...before,...page.filter(row=>!seen.has(row.noteId))];
}
