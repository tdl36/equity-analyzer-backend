import {validWork} from './workspace-model.mjs';
export function historyPage(value,ticker='') {
 if(value?.readOnly!==true||!Array.isArray(value.assignments)||value.assignments.length>30||
    !(value.nextCursor===null||(typeof value.nextCursor==='string'&&value.nextCursor.length>0&&value.nextCursor.length<=512))) throw Error('Saved assignment page is incomplete. Retry this read.');
 const seen=new Set();
 const rows=value.assignments.map(row=>{
  if(!row||!validWork('stockanalysis','stock-analysis','assignment:'+row.id)||
     typeof row.ticker!=='string'||! /^[A-Z0-9][A-Z0-9.\-]{0,29}$/.test(row.ticker)||
     (ticker&&row.ticker!==ticker)||typeof row.status!=='string'||row.status.length>40||seen.has(row.id)) throw Error('Saved assignments do not match this archive. Reload its first page.');
  seen.add(row.id);return row;
 });
 return {rows,cursor:value.nextCursor};
}
export function appendHistory(before,page) {
 const seen=new Set(before.map(row=>row.id));
 return [...before,...page.filter(row=>!seen.has(row.id))];
}
