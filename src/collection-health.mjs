export function collectionHealth(snapshot,now=Date.now()){
 const policies=snapshot?.policies||[],requests=snapshot?.requests||[];
 const active=requests.filter(r=>!['complete','cancelled'].includes(r.status));
 const waiting=active.filter(r=>r.status==='queued');
 const oldest=waiting.length?Math.max(...waiting.map(r=>Math.max(0,(now-r.created*1000)/60000))):0;
 return {scheduled:policies.filter(p=>p.enabled&&p.hours>0).length,verified:policies.filter(p=>p.lastSuccess).length,total:policies.length,
 pending:active.length,needsAttention:active.filter(r=>['needs_auth','attention'].includes(r.status)).length,
 overdue:waiting.filter(r=>now-r.created*1000>30*60000).length,oldestMinutes:Math.floor(oldest)};
}
