// Freeze both the requested wording and the context that informed it.
export function rewriteScope(fields, scope) {
 const selected=scope==='*'?fields:fields.filter(f=>f.id===scope);
 return {fields:selected,contextFields:scope==='*'?[]:fields.filter(f=>f.id!==scope)};
}
export function rewriteStillMatches(body, fields) {
 const current=new Map(fields.map(f=>[f.id,f]));
 const frozen=[...body.fields,...(body.contextFields||[])];
 return frozen.every(f=>JSON.stringify(current.get(f.id))===JSON.stringify(f)) &&
  (!Object.hasOwn(body,'contextFields')||frozen.length===fields.length);
}
