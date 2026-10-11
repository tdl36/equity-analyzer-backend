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
