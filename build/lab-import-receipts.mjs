const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const uuid=v=>typeof v==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(v);
const fail=()=>{throw Error('Import status or exact-byte receipts could not be verified. Existing progress is retained; refresh status before starting another import.');};
export function checkedLabImport(job,ticker,request=null){
 if(!object(job)||!uuid(job.requestId)||!object(job.input)||job.input.ticker!==ticker||!['thesis','note'].includes(job.input.task)||!Number.isSafeInteger(job.input.revision)||job.input.revision<1||!['queued','frozen','attention','complete','cancelled'].includes(job.status)||job.researchStarted!==false)fail();
 if(request&&(job.requestId!==request.requestId||job.input.task!==request.task||job.input.revision!==request.revision))fail();
 const refs=job.references;
 if(!Array.isArray(refs)||!refs.length||refs.length>1000||new Set(refs).size!==refs.length||refs.some(r=>typeof r!=='string'||!r.startsWith('icloud:')||r.length>1200))fail();
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

export function checkedLabImportRequest(body){
 if(!object(body)||!uuid(body.requestId)||!['thesis','note'].includes(body.task)||!Number.isSafeInteger(body.revision)||body.revision<1||Object.keys(body).some(k=>!['requestId','task','revision'].includes(k)))throw Error('Saved import request is unreadable. Preserve this session and inspect import status.');
 return body;
}
export async function reconcileLabImport(read,api,ticker,body){
 checkedLabImportRequest(body);
 try{return checkedLabImport(await read(api,`/api/pipeline-lab/icloud-import/${body.requestId}`),ticker,body);}
 catch(e){if(e.status===404)return null;throw e;}
}

export function checkedResearchImport(receipt,ticker,request,expectedSources=null){
 checkedLabImportRequest(request);
 const reject=()=>{throw Error('Research import receipt could not be verified. The saved request is retained; reconcile the same request before importing again.');};
 if(!object(receipt)||receipt.requestId!==request.requestId||receipt.input?.ticker!==ticker||receipt.input?.task!==request.task||receipt.input?.revision!==request.revision||receipt.revision!==request.revision+1||receipt.researchStarted!==false)reject();
 const rows=receipt.receipts;
 if(!Array.isArray(rows)||!rows.length||rows.length>1000||new Set(rows.map(r=>r?.source)).size!==rows.length||rows.some(r=>!object(r)||typeof r.source!=='string'||!/^charlie:research_[a-zA-Z0-9_-]{1,128}$/.test(r.source)||typeof r.target!=='string'||!/^charlie:thesis_[1-9][0-9]*$/.test(r.target)||typeof r.sha256!=='string'||!/^[a-f0-9]{64}$/.test(r.sha256)||!Number.isSafeInteger(r.bytes)||r.bytes<1||r.bytes>15000000)||rows.reduce((n,r)=>n+r.bytes,0)>60000000)reject();
 if(expectedSources!==null){const expected=expectedSources.filter(r=>r.startsWith('charlie:research_'));if(expected.length!==rows.length||rows.some(r=>!expected.includes(r.source)))reject();}
 return receipt;
}
