export function editionSelection(pack, corrections, ids, requestId) {
 if(!pack||pack.status!=='pending_only'||!Array.isArray(ids)||!ids.length||ids.length>20||new Set(ids).size!==ids.length)throw Error('Choose between one and twenty different corrections.');
 const paths=new Set();
 for(const id of ids){const row=corrections.find(c=>c.id===id);if(!row||row.stale||row.fingerprint!==pack.fingerprint)throw Error('A selected correction changed or is stale. Reload and choose again.');if(paths.has(row.proposal.path))throw Error('Choose only one correction for each claim.');paths.add(row.proposal.path);}
 return {requestId,assignmentId:pack.assignmentId,runId:pack.runId,fingerprint:pack.fingerprint,fundamentalsDigest:pack.fundamentalsDigest,parentId:pack.parentId,correctionIds:[...ids]};
}
export const claimLabel = path => path.replace(/^\//,'').split('/').map(x=>/^\d+$/.test(x)?String(Number(x)+1):x.replaceAll('_',' ')).join(' · ');
export const basisLabel = value => ({reported_fact:'Reported fact',management_guidance:'Management guidance',broker_estimate:'Broker estimate',interpretation:'Interpretation',hypothesis:'Hypothesis'}[value]||'Claim type unavailable');
