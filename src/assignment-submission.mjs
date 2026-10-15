// Persist before sending. A saved intent is not proof that the server received it.
export function submissionKey(api,ticker){return 'charlie-assignment-submit-v1:'+encodeURIComponent(api)+':'+encodeURIComponent(ticker);}
export function readSubmission(storage,key,ticker){
 const raw=storage.getItem(key);if(!raw)return null;
 let value;try{value=JSON.parse(raw);}catch{throw Error('Saved assignment request is unreadable. Keep this browser session and inspect saved assignments before starting again.');}
 if(!value||value.version!==1||!value.body||value.body.ticker!==ticker||typeof value.body.requestId!=='string'||!value.body.requestId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/))throw Error('Saved assignment request is invalid. Inspect saved assignments before starting again.');
 return value.body;
}
export function reserveSubmission(storage,key,body){
 const existing=readSubmission(storage,key,body.ticker);
 if(existing)return existing;
 storage.setItem(key,JSON.stringify({version:1,body}));
 const saved=readSubmission(storage,key,body.ticker);
 if(JSON.stringify(saved)!==JSON.stringify(body))throw Error('Could not preserve the submission identity. No request was sent.');
 return saved;
}
export function clearSubmission(storage,key,body){
 const saved=readSubmission(storage,key,body.ticker);
 if(saved&&saved.requestId!==body.requestId)throw Error('A different assignment request is pending. Reload before continuing.');
 storage.removeItem(key);
}
// Compare frozen user choices, not server-added execution state. Keep an ambiguous
// intent until both its identity and normalized plan are present in the receipt.
const stable=value=>JSON.stringify(value,(_,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])):v);
function receiptPlan(value,receipt=false){
 if(!value||typeof value!=='object'||!Array.isArray(value.outputs)||typeof value.ticker!=='string'||typeof value.since!=='string'||typeof value.until!=='string')return null;
 if(receipt&&(typeof value.horizon!=='string'||typeof value.instruction!=='string'))return null;
 if(value.outputs.some(x=>typeof x!=='string')||new Set(value.outputs).size!==value.outputs.length)return null;
 if(['visualFundamentals','collectionOnly'].some(k=>value[k]!==undefined&&typeof value[k]!=='boolean'))return null;
 const out={ticker:value.ticker.trim().toUpperCase(),since:value.since,until:value.until,outputs:[...value.outputs].sort(),horizon:(value.horizon??'12–24 months').trim(),instruction:(value.instruction??'').trim(),visualFundamentals:value.visualFundamentals??false,collectionOnly:value.collectionOnly??false};
 for(const field of ['meeting','origin','labSelection','labFullNote'])if(value[field]!==undefined)out[field]=value[field];
 return out;
}
export function matchingReceipt(data,body){
 try{
  if(data?.id!==body.requestId||data.ticker!==body.ticker.trim().toUpperCase())return false;
  const saved=receiptPlan(data.input,true),requested=receiptPlan(body);
  return !!saved&&!!requested&&stable(saved)===stable(requested);
 }catch{return false;}
}
