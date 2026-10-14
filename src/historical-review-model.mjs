const baselineFields = ['assignmentId','parentEditionId','fundamentalsDigest','sourcePackFingerprint','parentEditionDigest'];
function matchingBaselines(values, assignmentId, editionId) {
  const first = values[0];
  return values.every(value => value && value.assignmentId === assignmentId && value.parentEditionId === editionId &&
    baselineFields.every(key => typeof value[key] === 'string' && value[key] === first[key]) &&
    baselineFields.slice(2).every(key => /^[a-f0-9]{64}$/.test(value[key])));
}

export function historicalInventory(sources, retrievals, reviews, assignmentId, editionId) {
  const matches = value => value?.assignmentId === assignmentId && value?.parentEditionId === editionId;
  if (!matchingBaselines([sources?.baseline,retrievals?.baseline,reviews?.baseline],assignmentId,editionId) || !matches(retrievals) || !matches(reviews) ||
      [sources,retrievals,reviews].some(value=>value?.valuationReady !== false || typeof value.truncated !== 'boolean') ||
      !Array.isArray(sources.originals) || sources.originals.length > 60 ||
      !Array.isArray(retrievals.retrievals) || retrievals.retrievals.length > 60 ||
      !Array.isArray(reviews.reviews) || reviews.reviews.length > 60 ||
      sources.originals.some(row=>!row || !['daily_prices','sec_companyfacts'].includes(row.kind) || typeof row.id !== 'string') ||
      retrievals.retrievals.some(row=>!row || !['evidence_retained','unresolved'].includes(row.status) || row.retryAllowed !== false) ||
      reviews.reviews.some(row=>!row || row.status !== 'pending' || row.requiresEvidenceRecheck !== true)) {
    throw Error('Historical evidence does not match this edition or has an unexpected status. Reload its saved history.');
  }
  return {sources,retrievals,reviews};
}

export function historicalReview(value, assignmentId, editionId, reviewId) {
  const snapshot = value?.snapshot, payload = snapshot?.payload;
  if (value?.id !== reviewId || value?.status !== 'pending' || payload?.status !== 'pending' ||
      payload?.baseline?.assignmentId !== assignmentId || payload?.baseline?.parentEditionId !== editionId ||
      payload?.diagnosis?.valuationReady !== false || !Array.isArray(payload?.diagnosis?.observations) ||
      payload.diagnosis.observations.length > 30 || !/^[a-f0-9]{64}$/.test(snapshot?.reviewDigest || '')) {
    throw Error('This historical review does not match the selected edition. Reload its saved history.');
  }
  const proposals = payload.proposal?.observations;
  if (!Array.isArray(proposals) || proposals.length !== payload.diagnosis.observations.length) {
    throw Error('Historical review inputs are incomplete. Reopen its saved version.');
  }
  const text = value => typeof value === 'string' && value.length <= 500 ? value : 'Not established';
  return {id:reviewId, observations:payload.diagnosis.observations.map((row,index) => {
    if (!row || row.index !== index) throw Error('Historical evidence checks no longer match their saved inputs.');
    const original = proposals[index];
    return {index, message:typeof row.message === 'string' ? row.message : 'Evidence remains unverified.',
      price:text(original?.price?.value), priceDate:text(original?.price?.date),
      priceBasis:original?.price?.basis === 'unadjusted_close' ? 'Unadjusted closing price' : text(original?.price?.basis), earnings:text(original?.earnings?.value),
      earningsBasis:text(original?.earnings?.basis), periodStart:text(original?.earnings?.periodStart),
      periodEnd:text(original?.earnings?.periodEnd), publishedDate:text(original?.earnings?.publishedDate)};
  })};
}

export function historicalCandidate(value, selection) {
  const p=value?.payload, expected=selection.observation;
  if (!/^[a-f0-9]{64}$/.test(value?.candidateDigest||'') || p?.version!==1 ||
      !matchingBaselines([selection.baseline,p?.baseline],selection.baseline.assignmentId,selection.baseline.parentEditionId) ||
      p?.valuationReady!==false || p?.investorReviewRequired!==true ||
      !Array.isArray(p?.unresolvedChecks) ||
      !['security_currency_mapping','corporate_actions','filing_inventory_completeness','source_provenance','investor_review'].every(k=>p.unresolvedChecks.includes(k)) ||
      p?.selections?.length!==1 || p?.observations?.length!==1) throw Error('Candidate inputs do not match this frozen review. Reload saved evidence.');
  const saved=p.selections[0], row=p.observations[0];
  if(saved?.date!==expected.date || row?.date!==expected.date ||
     !['priceSource','earningsSource'].every(k=>saved?.[k]?.id===expected[k].id && saved?.[k]?.sha256===expected[k].sha256) ||
     !['incomplete','inputs_extracted'].includes(row?.status) || !Array.isArray(row?.gaps) ||
     row.gaps.length>2 || row.gaps.some(g=>!g || !['price','earnings'].includes(g.input) || !['missing_original','wrong_original_kind','unverified_original'].includes(g.code))) throw Error('Historical input selection changed. Reload saved evidence.');
  if(row.status==='inputs_extracted' && (row.gaps.length || typeof row.price?.value!=='string' || typeof row.earnings?.value!=='string')) throw Error('Historical inputs are incomplete. Reload saved evidence.');
  const text=v=>typeof v==='string'&&v.length<=500?v:'Not established';
  return {date:row.date,price:text(row.price?.value),earnings:text(row.earnings?.value),
    periodEnd:text(row.earnings?.periodEnd),publishedDate:text(row.earnings?.publishedDate),
    gaps:row.gaps.map(g=>`${g.input==='price'?'Price':'Earnings'}: ${g.message?text(g.message):'The selected original is missing or has the wrong type.'}`)};
}

export function historicalPlan(value, baseline, date) {
 const p=value?.payload;
 if(!/^[a-f0-9]{64}$/.test(value?.planDigest||'') || p?.version!==1 || p?.date!==date ||
    !matchingBaselines([baseline,p?.baseline],baseline.assignmentId,baseline.parentEditionId) ||
    p?.valuationReady!==false || p?.investorReviewRequired!==true ||
    !['selected','ambiguous','needs_evidence'].includes(p?.status) || !Array.isArray(p?.gaps) || p.gaps.length>2 ||
    !Array.isArray(p?.checks) || p.checks.length>60 || !Array.isArray(p?.inventory) || p.inventory.length>60 ||
    p.checks.some(c=>!c || !['price','earnings'].includes(c.input) || typeof c.eligible!=='boolean' || typeof c.source?.id!=='string') ||
    p.inventory.some(r=>!r || typeof r.id!=='string' || !/^[a-f0-9]{64}$/.test(r.sha256||'')))
   throw Error('Historical coverage does not match this frozen edition. Reload evidence.');
 if(p.status==='selected') {
  const observation=p.candidate?.payload?.selections?.[0];
  if(p.gaps.length || observation?.date!==date || !['price','earnings'].every(part=>{
   const receipt=observation[part+'Source'];
   return p.checks.filter(c=>c.input===part && c.eligible===true).length===1 && p.checks.some(c=>c.input===part && c.eligible===true && c.source?.id===receipt?.id && c.source?.sha256===receipt?.sha256) && p.inventory.some(r=>r.id===receipt?.id && r.sha256===receipt?.sha256);
  })) throw Error('Selected originals do not match the retained coverage. Reload evidence.');
  return {value:historicalCandidate(p.candidate,{baseline,observation})};
 }
 if(p.candidate || !p.gaps.length || p.gaps.some(g=>!g || !['price','earnings'].includes(g.input) || !['missing_coverage','ambiguous_receipts'].includes(g.code))) throw Error('Historical coverage is incomplete or inconsistent. Reload evidence.');
 return {coverage:p.gaps.map(g=>`${g.input==='price'?'Price':'Earnings'}: ${g.code==='missing_coverage'?'No retained original covers this exact date.':'Multiple retained originals qualify. Choose the originals explicitly below.'}`)};
}
