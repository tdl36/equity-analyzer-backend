import {parseTimestamp} from './workspace-model.mjs';
export const collectionTime=value=>parseTimestamp(typeof value==='number'?value*1000:value);
export const collectionOpen=row=>!['complete','cancelled'].includes(row.status);
export function collectionRequests(rows=[],query='',showFinished=false) {
 const q=query.trim().toUpperCase();
 const rank=r=>['attention','needs_auth'].includes(r.status)?0:collectionOpen(r)?1:2;
 return rows.filter(r=>(showFinished||collectionOpen(r))&&(!q||String(r.ticker||'').toUpperCase().includes(q)))
  .sort((a,b)=>rank(a)-rank(b)||(collectionTime(b.created)?.getTime()||0)-(collectionTime(a.created)?.getTime()||0));
}
export function collectionNext(row) {
 if(row.status==='needs_auth')return 'Complete sign-in or MFA directly in AlphaSense on the Mac, then resume this request.';
 if(row.status==='attention')return 'Resolve the saved issue before retrying. The frozen company and source scope will stay unchanged.';
 if(row.status==='queued')return 'Waiting for the browser worker. Keep the Mac awake, online, and AlphaSense signed in.';
 if(row.status==='running'||row.status==='collecting')return 'The worker owns this collection. Inspect saved search progress before submitting another request.';
 if(row.status==='complete')return 'Collection is complete. Check the parent assignment for research and saved outputs.';
 if(row.status==='cancelled')return 'Collection was cancelled. Existing originals and saved progress remain available.';
 return 'Inspect the saved state before taking action; no completion is inferred.';
}
// Queue metadata describes a read, not observed browser progress or an ETA.
export function collectionQueueMessage(row, now=Date.now()) {
 const q=row.queueState;
 if(!q)return 'Queue eligibility has not been reported by the Mac yet. A worker check-in alone does not prove this request has started.';
 if(!Number.isFinite(q.asOf)||q.asOf*1000>now||now-q.asOf*1000>180000||q.requestStatus!==row.status)return 'Queue detail is stale or inconsistent. Wait for a fresh Mac report before interpreting eligibility or leases.';
 const messages={leased:'This request holds the collection slot. An active lease is not proof of a completed search or download.',needs_attention:'This request is excluded from automatic claims until its saved issue is resolved and it is resumed.',missing_policy:'The ticker policy is missing, so the worker cannot claim this request. Inspect collection configuration; do not submit a duplicate.',policy_paused:'This request is excluded because its ticker policy is paused and it has no manual collection override.',lease_wait:'This request has an unexpired lease and cannot be claimed again yet.',not_claimable:'This saved state is not eligible for a browser claim. Inspect its saved issue before changing anything.'};
 let message=messages[q.state];
 if(['eligible','waiting_for_slot'].includes(q.state)){
  if(!Number.isSafeInteger(q.defaultPosition)||q.defaultPosition<1||!Number.isSafeInteger(q.eligibleCount)||q.eligibleCount<q.defaultPosition)return 'Queue position is unavailable. Inspect the next Mac report; no start time is inferred.';
  message=`Eligible for collection · default queue position ${q.defaultPosition} of ${q.eligibleCount}.`;
  if(q.state==='waiting_for_slot'){
   if(!Array.isArray(q.active)||!q.active.length||q.active.some(r=>typeof r.ticker!=='string'||!Number.isFinite(r.leaseUntil)||r.leaseUntil<=q.asOf))return 'Collection slot details are unavailable. Wait for a fresh Mac report.';
   const names=q.active.map(r=>r.ticker).join(', ');
   message+=` Waiting while ${names}${q.activeCount>q.active.length?' and other requests':''} hold the collection slot.`;
   if(q.active.some(r=>r.leaseUntil*1000<=now))message+=' A reported lease has since expired; refresh before assuming the slot is free.';
  }else message+=' No active collection lease was reported; the next browser-worker check must still claim it.';
  message+=' This is the default order, not an estimated start time; explicitly targeted worker requests may differ.';
 }
 if(!message)return 'Queue state is unrecognized. Inspect the next Mac report before retrying.';
 return message+` Queue checked ${new Date(q.asOf*1000).toLocaleTimeString()}.`;
}
