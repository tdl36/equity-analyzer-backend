export function editionSelection(pack, corrections, ids, requestId) {
 if(!pack||pack.status!=='pending_only'||!Array.isArray(ids)||!ids.length||ids.length>60||new Set(ids).size!==ids.length)throw Error('Choose between one and sixty different corrections.');
 const paths=new Set();
 for(const id of ids){const row=corrections.find(c=>c.id===id);if(!row||row.stale||row.fingerprint!==pack.fingerprint)throw Error('A selected correction changed or is stale. Reload and choose again.');if(paths.has(row.proposal.path))throw Error('Choose only one correction for each claim.');paths.add(row.proposal.path);}
 return {requestId,assignmentId:pack.assignmentId,runId:pack.runId,fingerprint:pack.fingerprint,fundamentalsDigest:pack.fundamentalsDigest,parentId:pack.parentId,correctionIds:[...ids]};
}
export const claimLabel = path => path.replace(/^\//,'').split('/').map(x=>/^\d+$/.test(x)?String(Number(x)+1):x.replaceAll('_',' ')).join(' · ');
export const basisLabel = value => ({reported_fact:'Reported fact',management_guidance:'Management guidance',broker_estimate:'Broker estimate',interpretation:'Interpretation',hypothesis:'Hypothesis'}[value]||'Claim type unavailable');

export function editionReviewState(edition) {
 const action=edition.decision?.action;
 if(action==='accept')return {label:edition.currentAccepted?'Accepted edition · current review version':'Accepted edition · prior review version',canPreview:false};
 if(action==='reject')return {label:'Rejected edition',canPreview:false};
 return {label:'Pending edition',canPreview:!edition.stale&&!edition.decision&&(!edition.status||edition.status==='pending')};
}

export function verifiedEditionPreview(value, editionId, assignmentId, runId, ticker) {
 if(!value||value.id!==editionId||value.assignmentId!==assignmentId||value.runId!==runId||value.ticker!==ticker||value.status!=='pending'||!['illustrated','brief'].includes(value.layout)||typeof value.html!=='string'||!value.html.trim())throw Error('The preview does not match this pending edition. Reload the saved assignment.');
 return value;
}

export function editionDecision(edition, pack, action, requestId) {
 if(!pack?.canDecide||edition.stale||edition.decision||edition.record.parentId!==pack.parentId||!['accept','reject'].includes(action))throw Error('Reload the edition and sign in as the investor before deciding.');
 const r=edition.record;
 return {requestId,editionId:edition.id,action,...Object.fromEntries(['assignmentId','runId','fingerprint','fundamentalsDigest','parentId','correctionIds','selectionDigest'].map(k=>[k,r[k]]))};
}
export function removedEditionClaims(edition, pack) {
 return (pack.acceptedEdition?.claims||[]).filter(c=>!edition.record.correctionIds.includes(c.correctionId));
}
