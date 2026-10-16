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
