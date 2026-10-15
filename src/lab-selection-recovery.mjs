export function selectionRecoveryKey(api,ticker){return `charlie.lab.selection.v1:${api}:${ticker}`;}
export function validSelectionDraft(value,ticker){
 return value?.version===1&&value.ticker===ticker&&Number.isInteger(value.revision)&&value.revision>=0&&
 ['thesis','note'].every(task=>Array.isArray(value.selection?.[task])&&value.selection[task].length<=1000&&value.selection[task].every(k=>typeof k==='string'&&k.length>0&&k.length<=1000)&&new Set(value.selection[task]).size===value.selection[task].length);
}
export function readSelectionDraft(storage,key,ticker){
 const raw=storage.getItem(key);if(!raw)return null;
 let value;try{value=JSON.parse(raw);}catch{throw Error('Saved source-selection recovery is unreadable. Your central settings are unchanged.');}
 if(!validSelectionDraft(value,ticker))throw Error('Saved source-selection recovery is invalid. Your central settings are unchanged.');
 return value;
}
export function writeSelectionDraft(storage,key,ticker,revision,selection){
 const value={version:1,ticker,revision,selection};
 if(!validSelectionDraft(value,ticker))throw Error('Source selections could not be retained for recovery.');
 const raw=JSON.stringify(value);storage.setItem(key,raw);
 if(storage.getItem(key)!==raw)throw Error('Source-selection recovery storage could not be confirmed.');
}
