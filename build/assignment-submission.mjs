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
export function matchingReceipt(data,body){return data?.id===body.requestId;}
