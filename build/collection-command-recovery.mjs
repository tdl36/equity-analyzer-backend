const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const uuid=v=>typeof v==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(v);
const actions=new Set(['save','trigger','save_batch','cancel','retry','event_refresh','save_trigger']);
export const commandRecoveryKey=api=>'charlie-collection-command:'+api;
export function readCommand(storage,key){
 const raw=storage.getItem(key);if(raw===null)return null;
 if(raw.length>500000)throw Error('Collection command recovery is too large. Restore browser storage before sending another command.');
 let value;try{value=JSON.parse(raw);}catch{throw Error('Collection command recovery is unreadable. Restore browser storage before sending another command.');}
 if(!object(value)||Object.keys(value).sort().join(',')!=='action,payload,requestId'||!uuid(value.requestId)||!actions.has(value.action)||!object(value.payload))throw Error('Collection command recovery is invalid. Restore browser storage before sending another command.');
 return value;
}
export function retainCommand(storage,key,body){
 const prior=readCommand(storage,key),raw=JSON.stringify(body);
 if(prior&&JSON.stringify(prior)!==raw)throw Error('Another collection command remains unresolved. Check its status before sending new work.');
 storage.setItem(key,raw);
 if(storage.getItem(key)!==raw)throw Error('Collection recovery could not be saved. No command was sent.');
 readCommand(storage,key);
}
export function clearCommand(storage,key,body){
 const prior=readCommand(storage,key);
 if(prior&&JSON.stringify(prior)!==JSON.stringify(body))throw Error('Collection recovery changed. Reload before sending another command.');
 storage.removeItem(key);
 if(storage.getItem(key)!==null)throw Error('The command was recorded but browser recovery could not be cleared. Check status before sending another command.');
}
export function commandReceipt(row,body){
 if(!object(row)||row.id!==body.requestId||!['queued','applied','failed'].includes(row.status))throw Error('Collection command receipt could not be verified. Check its status before retrying.');
 return row;
}
export function exactCommand(data,body){
 if(!object(data)||!Array.isArray(data.commands)||data.commands.length>1)throw Error('Exact collection command status is unavailable. Retry the status check.');
 return data.commands.length?commandReceipt(data.commands[0],body):null;
}
