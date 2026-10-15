import {labDocuments} from './pipeline-lab-model.mjs';
export function collectedSources(data,job){
 const receipts=job.result.sources,rows=data?.documents;
 const fail=()=>{throw Error('Collected source inventory does not match this collection. Refresh before selecting evidence.');};
 if(data?.collectionId!==job.id||data?.ticker!==job.ticker||!Array.isArray(rows)||!Array.isArray(receipts)||rows.length!==receipts.length||rows.length>1000)fail();
 const expected=new Map(receipts.map(r=>[r.filename,r.sha256]));
 if(expected.size!==receipts.length||new Set(rows.map(r=>r?.filename)).size!==rows.length)fail();
 for(const r of rows){
  if(!r||typeof r.filename!=='string'||!expected.has(r.filename)||r.receiptSha256!==expected.get(r.filename)||!(/^[0-9a-f]{64}$/).test(r.receiptSha256)||
   !['available','restricted','missing','ambiguous'].includes(r.availability)||
   (['available','restricted'].includes(r.availability)?typeof r.id!=='string'||!/^thesis_[1-9][0-9]{0,18}$/.test(r.id):r.id!==null)||
   ['publisher','author','sourceKind'].some(k=>r[k]!=null&&typeof r[k]!=='string')||(r.publishedDate!=null&&typeof r.publishedDate!=='string'))fail();
 }
 const ids=rows.filter(r=>r.id).map(r=>r.id);if(new Set(ids).size!==ids.length)fail();
 return rows.map(r=>({...labDocuments([{...r,id:r.id||'unavailable:'+r.filename}],[])[0],id:r.id,date:r.publishedDate,dateLabel:'Published'}));
}
export function duplicateSelections(rows,selected){
 const hashes=new Set(rows.filter(r=>selected.includes(r.filename)).map(r=>r.receiptSha256));
 return new Set(rows.filter(r=>!selected.includes(r.filename)&&hashes.has(r.receiptSha256)).map(r=>r.filename));
}
