const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const uuid=v=>typeof v==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(v);
const fail=()=>{throw Error('Import status or exact-byte receipts could not be verified. Existing progress is retained; refresh status before starting another import.');};
export function checkedLabImport(job,ticker,request=null){
 if(!object(job)||!uuid(job.requestId)||!object(job.input)||job.input.ticker!==ticker||!['thesis','note'].includes(job.input.task)||!Number.isSafeInteger(job.input.revision)||job.input.revision<1||!['queued','frozen','attention','complete','cancelled'].includes(job.status)||job.researchStarted!==false)fail();
 if(request&&(job.requestId!==request.requestId||job.input.task!==request.task||job.input.revision!==request.revision))fail();
 const refs=job.references;
 if(!Array.isArray(refs)||!refs.length||refs.length>8||new Set(refs).size!==refs.length||refs.some(r=>typeof r!=='string'||!r.startsWith('icloud:')||r.length>1200))fail();
 if(job.manifest!==undefined||['frozen','complete'].includes(job.status)){
  const rows=job.manifest;
  if(!Array.isArray(rows)||rows.length!==refs.length||new Set(rows.map(r=>r?.reference)).size!==refs.length||rows.some(r=>!object(r)||!refs.includes(r.reference)||typeof r.filename!=='string'||!r.filename||typeof r.sha256!=='string'||!(/^[0-9a-f]{64}$/).test(r.sha256)||!Number.isSafeInteger(r.bytes)||r.bytes<1||r.bytes>15000000)||rows.reduce((n,r)=>n+r.bytes,0)>60000000)fail();
 }
 if(job.status==='complete'){
  if(job.completedRevision!==job.input.revision+1||!Array.isArray(job.receipts)||job.receipts.length!==refs.length||new Set(job.receipts.map(r=>r?.reference)).size!==refs.length)fail();
  for(const receipt of job.receipts){
   const source=job.manifest.find(r=>r.reference===receipt?.reference);
   if(!source||receipt.filename!==source.filename||receipt.sha256!==source.sha256||receipt.bytes!==source.bytes||typeof receipt.target!=='string'||!/^charlie:thesis_[1-9][0-9]*$/.test(receipt.target))fail();
  }
 }
 return job;
}
export function checkedLabImports(data,ticker){
 if(!Array.isArray(data?.imports)||data.imports.length>25)fail();
 const rows=data.imports.map(job=>checkedLabImport(job,ticker));
 if(new Set(rows.map(r=>r.requestId)).size!==rows.length)fail();
 return rows;
}
