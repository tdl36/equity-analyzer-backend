import test from 'node:test';
import assert from 'node:assert/strict';
import {historicalEditionReadiness as check} from '../../src/historical-review-model.mjs';
const baseline={assignmentId:'a',parentEditionId:'e',fundamentalsDigest:'a'.repeat(64),sourcePackFingerprint:'b'.repeat(64),parentEditionDigest:'c'.repeat(64)};
const h='d'.repeat(64),day='2024-12-31';
function fixture(){return {valuationReady:false,dispatchAllowed:false,datePolicy:{datePolicyDigest:h,payload:{version:1,policy:'exact_fiscal_period_end_v1',baseline,asOf:'2025-03-04',dates:[day],anchors:[{date:day,periodEnd:day,periodStart:'2024-01-01',basis:'fiscal_period_end'}],gaps:[],status:'dates_available',tradingDatesVerified:false,valuationReady:false}},readiness:{readinessDigest:h,payload:{version:1,baseline,dates:[day],valuationReady:false,dispatchAllowed:false,investorReviewRequired:true,status:'needs_evidence_review',unresolvedChecks:['security_currency_mapping','corporate_actions','filing_inventory_completeness','source_provenance','investor_review'],observations:[{date:day,planDigest:h,status:'needs_evidence',gaps:[{input:'price',code:'missing_coverage'}]}]}}};}
test('missing coverage and unique inputs stay unapproved',()=>{
 const v=fixture();assert.match(check(v,baseline).rows[0].message,/missing/);
 v.readiness.payload.status='inputs_selected';Object.assign(v.readiness.payload.observations[0],{status:'selected',gaps:[],candidateDigest:h,selection:{date:day,priceSource:{id:'p',sha256:h},earningsSource:{id:'e',sha256:h}}});
 assert.match(check(v,baseline).rows[0].message,/review remain/);
});
test('date review gap without candidate is explicit',()=>{
 const v=fixture();delete v.readiness;Object.assign(v.datePolicy.payload,{status:'needs_date_review',dates:[],anchors:[],gaps:[{code:'no_frozen_annual_periods'}]});
 assert.equal(check(v,baseline).needsDateReview,true);
});
test('mixed baseline, altered dates and unsafe statuses are rejected',()=>{
 for(const mutate of [v=>v.dispatchAllowed=true,v=>v.datePolicy.payload.tradingDatesVerified=true,v=>v.readiness.payload.baseline={...baseline,parentEditionDigest:h},v=>v.readiness.payload.dates=['2025-01-01'],v=>v.readiness.payload.observations=[],v=>v.readiness.payload.unresolvedChecks=[],v=>v.datePolicy.payload.anchors[0].date='2024-12-30',v=>v.readiness.payload.status='inputs_selected']){
  const v=fixture();mutate(v);assert.throws(()=>check(v,baseline));
 }
});
