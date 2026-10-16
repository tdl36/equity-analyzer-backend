export function checkedProcessingReadiness(value,proposal){
 if(!value||value.proposalId!==proposal.id||value.collectionId!==proposal.collectionId||value.ticker!==proposal.ticker||
    value.revision!==proposal.revision||value.digest!==proposal.digest||value.readOnly!==true||
    !['pending','cancelled','dispatched'].includes(value.status)||typeof value.ready!=='boolean'||
    !Array.isArray(value.blockers)||value.blockers.some(b=>typeof b!=='string'||!b.trim())||
    value.ready!==(value.status==='pending'&&value.blockers.length===0)||
    typeof value.checkedAt!=='string'||!Number.isFinite(Date.parse(value.checkedAt)))
  throw Error('Readiness does not match this saved proposal. Recheck before approving; no processing was requested.');
 return value;
}
