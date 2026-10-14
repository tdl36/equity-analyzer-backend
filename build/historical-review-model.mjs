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
