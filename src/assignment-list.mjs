// Reject an incomplete status response; never mistake it for an empty queue.
const record=value=>value!==null&&typeof value==='object'&&!Array.isArray(value);
export function assignmentRows(data,ticker){
 const rows=data?.assignments;
 if(!Array.isArray(rows)||rows.length>30||new Set(rows.map(row=>row?.id)).size!==rows.length||rows.some(row=>
  !record(row)||typeof row.id!=='string'||!row.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)||row.ticker!==ticker||
  !['queued','running','attention','complete','stopped','failed','cancelled'].includes(row.status)||
  !record(row.input)||!record(row.result)||typeof row.input.since!=='string'||typeof row.input.until!=='string'||
  (row.result.artifacts!=null&&(!Array.isArray(row.result.artifacts)||row.result.artifacts.some(a=>!record(a)||typeof a.kind!=='string'||typeof a.url!=='string')))
 ))throw Error('Assignment status could not be verified. Previously loaded assignments remain visible; starting new research is paused until status refresh succeeds.');
 return rows;
}
