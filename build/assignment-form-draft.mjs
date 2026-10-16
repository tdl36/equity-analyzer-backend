import {assignmentOutputs} from './assignment-defaults.mjs';
const canonical=v=>JSON.stringify(v,(_,x)=>x&&typeof x==='object'&&!Array.isArray(x)?Object.fromEntries(Object.keys(x).sort().map(k=>[k,x[k]])):x);
export function assignmentFormKey({api,ticker,origin=null,labSelection=null,labMode=false}){
 return 'charlie-assignment-form-v1:'+canonical({api,ticker,origin,labSelection,labMode});
}
const strings={purpose:30,eventTitle:200,eventDate:10,since:10,until:10,horizon:100,instruction:1800,meetingDate:10,meetingType:20};
const booleans=['collectionOnly','fullNote','notePdfText','visualFundamentals'];
const keys=[...Object.keys(strings),...booleans,'collectionLimit','chosen'].sort();
export function checkedAssignmentForm(value){
 if(!value||typeof value!=='object'||Array.isArray(value)||canonical(Object.keys(value).sort())!==canonical(keys)||
    Object.entries(strings).some(([k,n])=>typeof value[k]!=='string'||value[k].length>n)||booleans.some(k=>typeof value[k]!=='boolean')||
    !Number.isFinite(value.collectionLimit)||!Array.isArray(value.chosen)||new Set(value.chosen).size!==value.chosen.length||
    value.chosen.some(id=>!assignmentOutputs.some(([key])=>key===id))||!['thesis_review','event_note'].includes(value.purpose)||
    !['management','expert','conference','other'].includes(value.meetingType))throw Error('Browser form recovery is unreadable. It has been retained; clear it explicitly before keeping new choices here.');
 return {...value,chosen:[...value.chosen]};
}
export function readAssignmentForm(storage,key){
 const raw=storage.getItem(key);if(raw===null)return null;
 if(raw.length>16000)throw Error('Browser form recovery is too large to read. It has been retained.');
 let data;try{data=JSON.parse(raw);}catch{throw Error('Browser form recovery is unreadable. It has been retained.');}
 if(data?.version!==1||data.scope!==key)throw Error('Browser form recovery belongs to another context or version. It has been retained.');
 return checkedAssignmentForm(data.form);
}
export function writeAssignmentForm(storage,key,form){
 const value=checkedAssignmentForm(form);
 storage.setItem(key,JSON.stringify({version:1,scope:key,form:value}));
 if(canonical(readAssignmentForm(storage,key))!==canonical(value))throw Error('Browser form recovery could not be verified. Keep this screen open until your choices can be retained.');
}
export function clearAssignmentForm(storage,key){
 storage.removeItem(key);
 if(storage.getItem(key)!==null)throw Error('Browser form recovery could not be cleared. Keep this screen open.');
}
