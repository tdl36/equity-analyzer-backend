export const caseFields = {thesis:'Investment thesis',variantView:'Variant view',marketBaseline:'Market expectations',changeConditions:'Reconsideration conditions',scenarios:'Valuation scenarios'};
export const assumptionFields = {claim:'Claim',support:'Supporting evidence',contrary:'Contrary evidence',nextTest:'Next test',sourceReference:'Source reference',evidenceType:'Attribution'};
const value = x => typeof x === 'object' ? JSON.stringify(x ?? {}, null, 2) : String(x ?? '');
export function compareCases(before, after) {
 const changes=[];
 for (const [key,label] of Object.entries(caseFields)) if(value(before?.[key])!==value(after?.[key])) changes.push({label,before:value(before?.[key]),after:value(after?.[key])});
 const a=new Map((before?.assumptions||[]).map(x=>[x.id,x])), b=new Map((after?.assumptions||[]).map(x=>[x.id,x]));
 for(const id of new Set([...a.keys(),...b.keys()])) {
  const old=a.get(id),next=b.get(id);
  if(!old||!next) changes.push({label:next?'Assumption added':'Assumption removed',before:old?.claim||'',after:next?.claim||'',assumptionId:id});
  else for(const [key,label] of Object.entries(assumptionFields)) if(value(old[key])!==value(next[key])) changes.push({label:`${next.claim} · ${label}`,before:value(old[key]),after:value(next[key]),assumptionId:id});
 }
 return changes;
}
export function pillarState(previous,current,id,hasBaseline=true) {
 const a=previous?.assumptions?.find(x=>x.id===id),b=current?.assumptions?.find(x=>x.id===id);
 if(!hasBaseline)return b?'Baseline':'—';
 if(!a&&!b)return '—';if(!a)return 'Added';if(!b)return 'Removed';
 return Object.keys(assumptionFields).some(k=>value(a[k])!==value(b[k]))?'Revised':'Unchanged';
}
export function latestWork(rows) {
 const records=new Map();for(const row of rows)if(!records.has(row.id)||records.get(row.id).revision<row.revision)records.set(row.id,row);
 return [...records.values()];
}
