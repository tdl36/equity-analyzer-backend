export function checkedNoteProgress(value,id,ticker) {
 if(value?.noteId!==id||value.ticker!==ticker||value.readOnly!==true||
    !['draft','attention','queued','running'].includes(value.status)||
    !Number.isSafeInteger(value.batchCount)||value.batchCount<1||value.batchCount>8||
    ![value.revision,value.acceptedRevision].every(r=>r===null||Number.isSafeInteger(r)&&r>0)||
    !Array.isArray(value.batches)||value.batches.length>8||typeof value.modelKey!=='string')throw Error('The saved progress does not match this note and company. No recovery action is available.');
 let previous=0;
 for(const row of value.batches){
  if(!row||!Number.isSafeInteger(row.batch)||row.batch<=previous||row.batch>value.batchCount||!['complete','dispatched','uncertain'].includes(row.status))throw Error('Saved batch outcomes cannot be verified. No recovery action is available.');
  previous=row.batch;
 }
 return value;
}
export const batchOutcome=status=>({complete:'Reply saved',dispatched:'Dispatch recorded · outcome not yet confirmed',uncertain:'Outcome uncertain · do not repeat'}[status]||'No dispatch recorded');
