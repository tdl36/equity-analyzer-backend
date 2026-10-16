// Read-only identity checks. Never repairs a response or infers acceptance.
const revisionNumber=value=>Number.isSafeInteger(value)&&value>0;
export function checkedNote(value,id,ticker,revision) {
 if(!value||value.noteId!==id||value.ticker!==ticker||!revisionNumber(value.revision)||
    (revision!==undefined&&value.revision!==revision)||typeof value.title!=='string'||
    typeof value.markdown!=='string'||!Array.isArray(value.qualityIssues)||
    !value.qualityIssues.every(x=>typeof x==='string')||!Array.isArray(value.sourcePack))
  throw Error('The saved note does not match the requested company, note and revision. No other note was substituted.');
 const acceptance=value.acceptance;
 if(acceptance!==undefined&&acceptance!==null&&(!acceptance||acceptance.revision!==value.revision||
    typeof acceptance.acceptedAt!=='string'||!Number.isFinite(Date.parse(acceptance.acceptedAt))))
  throw Error('The saved acceptance does not match this note revision. Reload the exact saved note.');
 return value;
}
export function checkedNoteHistory(value,id,ticker,before=null) {
 if(!value||value.noteId!==id||!Array.isArray(value.versions)||value.versions.length>50)
  throw Error('Version history does not match this saved note. The open note is unchanged.');
 let previous=before??Infinity;
 for(const row of value.versions){
  if(!row||row.ticker!==ticker||!revisionNumber(row.revision)||row.revision>=previous||
     typeof row.title!=='string'||typeof row.createdAt!=='string')
   throw Error('Version history contains an unexpected revision. The open note is unchanged.');
  previous=row.revision;
 }
 if(value.nextBefore!==null&&(!revisionNumber(value.nextBefore)||!value.versions.length||value.nextBefore!==previous))
  throw Error('Version history has an invalid continuation. The open note is unchanged.');
 return value;
}

// Confirm the immutable edit receipt and its exact saved text before clearing recovery.
export async function confirmedNoteEdit({post,read,id,ticker,body}) {
 let receipt;
 try { receipt=await post(`/api/pipeline-lab/notes/${encodeURIComponent(id)}/revisions`,body); }
 catch(error) {
  // Only a definitive rejection of the write releases its frozen submission.
  // A read failure after a successful write must retain that same identity.
  const failure=new Error(error.message);
  failure.editRejected=[400,409].includes(error.status);
  throw failure;
 }
 if(!receipt||receipt.noteId!==id||!revisionNumber(receipt.revision)||
    receipt.revision!==body.expectedRevision+1||receipt.status!=='draft'||typeof receipt.replayed!=='boolean')
  throw Error('The edit receipt does not confirm the submitted revision. Your frozen edit is retained for reconciliation.');
 const saved=checkedNote(await read(`/api/pipeline-lab/notes/${encodeURIComponent(id)}?revision=${receipt.revision}`),id,ticker,receipt.revision);
 if(saved.title!==body.title.trim()||saved.markdown!==body.markdown)
  throw Error('The saved revision does not match the submitted text. Your frozen edit is retained for reconciliation.');
 return saved;
}
