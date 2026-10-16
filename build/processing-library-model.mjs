// Bind metadata choices to the exact company, collection and requested page.
// These checks do not replace the server's original-byte checks at save/approval.
const less=(a,b)=>a.length<b.length||(a.length===b.length&&a<b);
const id=v=>typeof v==='string'&&/^[1-9][0-9]{0,18}$/.test(v)&&!less('9223372036854775807',v);
export function processingLibraryPage(data,{ticker,collectionId,after=null}){
 const fail=()=>{throw Error('Source library page could not be verified for this company and collection. Retry the read; your selections are retained.');};
 if(!data||data.ticker!==ticker||data.collectionId!==collectionId||data.after!==after||
  (after!==null&&!id(after))||!Array.isArray(data.documents)||data.documents.length>200||
  !(data.nextCursor===null||id(data.nextCursor)))fail();
 let previous=after;
 for(const row of data.documents){
  const value=typeof row?.id==='string'&&row.id.startsWith('thesis_')?row.id.slice(7):null;
  if(!id(value)||typeof row.filename!=='string'||!row.filename||typeof row.eligible!=='boolean'||(previous!==null&&!less(value,previous)))fail();
  previous=value;
 }
 if(data.nextCursor!==null&&(data.documents.length!==200||data.nextCursor!==previous))fail();
 return {documents:data.documents,nextCursor:data.nextCursor};
}

export function selectedLibraryMetadata(data,{ticker,collectionId,selected}){
 const fail=()=>{throw Error('Selected source metadata could not be verified. Retry the read; your selections are retained.');};
 const expected=[...selected].sort();
 if(!expected.length||expected.length>100||expected.some(v=>!id(v))||new Set(expected).size!==expected.length||
  !data||data.ticker!==ticker||data.collectionId!==collectionId||!Array.isArray(data.selectedIds)||
  JSON.stringify(data.selectedIds)!==JSON.stringify(expected)||!Array.isArray(data.missingIds)||
  data.missingIds.some(v=>!expected.includes(v))||new Set(data.missingIds).size!==data.missingIds.length)fail();
 const page=processingLibraryPage({...data,after:null,nextCursor:null},{ticker,collectionId});
 const found=page.documents.map(r=>r.id.slice(7)),all=[...found,...data.missingIds].sort();
 if(JSON.stringify(all)!==JSON.stringify(expected))fail();
 return {documents:page.documents,missingIds:data.missingIds};
}
