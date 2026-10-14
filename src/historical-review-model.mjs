import {acquisitionWarnings} from './historical-acquisition.mjs';
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
  return {acquisitionWarnings:acquisitionWarnings(p.acquisitionChecks,expected),date:row.date,price:text(row.price?.value),earnings:text(row.earnings?.value),
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

export function historicalEditionReadiness(value, baseline) {
  const policy=value?.datePolicy, p=policy?.payload, ready=value?.readiness, r=ready?.payload;
  const hash=x=>typeof x==='string' && /^[a-f0-9]{64}$/.test(x);
  const date=x=>typeof x==='string' && /^20\d{2}-\d{2}-\d{2}$/.test(x) && Number.isFinite(Date.parse(x)) && new Date(x).toISOString().slice(0,10)===x;
  const matches=x=>matchingBaselines([baseline,x],baseline?.assignmentId,baseline?.parentEditionId);
  const fail=()=>{throw Error('Saved date coverage does not match this edition. Reload its evidence.');};
  if(value?.valuationReady!==false || value?.dispatchAllowed!==false || !hash(policy?.datePolicyDigest) ||
     p?.version!==1 || p?.policy!=='exact_fiscal_period_end_v1' || !matches(p.baseline) ||
     p.tradingDatesVerified!==false || p.valuationReady!==false || !date(p.asOf) ||
     !Array.isArray(p.dates) || p.dates.length>30 || new Set(p.dates).size!==p.dates.length ||
     p.dates.some(d=>!date(d)||d>=p.asOf) || !Array.isArray(p.anchors) || p.anchors.length!==p.dates.length ||
     !Array.isArray(p.gaps) || p.gaps.length>30) fail();
  if(p.status==='needs_date_review') {
    if(p.dates.length || !p.gaps.length || ready) fail();
    return {needsDateReview:true,rows:[]};
  }
  if(p.status!=='dates_available'||!p.dates.length||p.gaps.length || !hash(ready?.readinessDigest) ||
     r?.version!==1||!matches(r.baseline)||r.valuationReady!==false||r.dispatchAllowed!==false||r.investorReviewRequired!==true||
     !Array.isArray(r.dates)||JSON.stringify(r.dates)!==JSON.stringify(p.dates)||
     !Array.isArray(r.observations)||r.observations.length!==p.dates.length ||
     !Array.isArray(r.unresolvedChecks)||!['security_currency_mapping','corporate_actions','filing_inventory_completeness','source_provenance','investor_review'].every(x=>r.unresolvedChecks.includes(x))) fail();
  const rows=r.observations.map((row,index)=>{
    const anchor=p.anchors[index],day=p.dates[index];
    if(anchor?.date!==day||anchor.periodEnd!==day||anchor.basis!=='fiscal_period_end'||!date(anchor.periodStart)||
       anchor.periodStart>=day||row?.date!==day||!hash(row.planDigest)||!Array.isArray(row.gaps)) fail();
    if(row.status==='selected') {
      const s=row.selection;
      if(row.gaps.length||!hash(row.candidateDigest)||s?.date!==day||
         ['priceSource','earningsSource'].some(k=>typeof s[k]?.id!=='string'||!s[k].id||!hash(s[k].sha256))) fail();
      return {date:day,message:'Inputs matched; valuation checks and investor review remain.'};
    }
    if(!['needs_evidence','ambiguous'].includes(row.status)||!row.gaps.length||row.selection||row.candidateDigest||
       row.gaps.some(g=>!['price','earnings'].includes(g?.input)||!['missing_coverage','ambiguous_receipts'].includes(g.code))) fail();
    return {date:day,message:row.gaps.map(g=>`${g.input==='price'?'Price':'Earnings'}: ${g.code==='missing_coverage'?'exact-date coverage missing':'multiple eligible originals require review'}`).join(' · ')};
  });
  const selected=r.observations.every(x=>x.status==='selected');
  if(r.status!==(selected?'inputs_selected':'needs_evidence_review')) fail();
  return {needsDateReview:false,rows};
}
