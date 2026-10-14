import test from 'node:test';
import assert from 'node:assert/strict';
import {historicalReview,historicalInventory} from '../../src/historical-review-model.mjs';

function fixture() {return {id:'review',status:'pending',snapshot:{reviewDigest:'a'.repeat(64),payload:{status:'pending',baseline:{assignmentId:'assignment',parentEditionId:'edition'},proposal:{observations:[{price:{value:'100.0000',date:'2025-03-03',basis:'unadjusted_close'},earnings:{value:'5',basis:'GAAP diluted EPS',periodStart:'2024-01-01',periodEnd:'2024-12-31',publishedDate:'2025-02-15'}}]},diagnosis:{valuationReady:false,observations:[{index:0,message:'Source amounts match; corporate actions remain unresolved.'}]}}}};}
test('historical review stays pending and retains qualifications',()=>{
  assert.match(historicalReview(fixture(),'assignment','edition','review').observations[0].message,/remain unresolved/);
});
test('saved amounts and vintages remain explicit without inventing a valuation',()=>{
  const row=historicalReview(fixture(),'assignment','edition','review').observations[0];
  assert.equal(row.price,'100.0000');assert.equal(row.publishedDate,'2025-02-15');assert.equal(row.earningsBasis,'GAAP diluted EPS');
  assert.equal(row.priceBasis,'Unadjusted closing price');
  assert.equal(row.multiple,undefined);
  const missing=fixture();missing.snapshot.payload.proposal.observations=[null];
  assert.equal(historicalReview(missing,'assignment','edition','review').observations[0].price,'Not established');
  const mismatch=fixture();mismatch.snapshot.payload.diagnosis.observations[0].index=2;
  assert.throws(()=>historicalReview(mismatch,'assignment','edition','review'));
});
test('wrong target or unexpected approval cannot be displayed as the selected pending review',()=>{
  for (const mutate of [v=>v.id='other',v=>v.status='accepted',v=>v.snapshot.payload.baseline.parentEditionId='other',v=>v.snapshot.payload.baseline.assignmentId='other',v=>v.snapshot.payload.diagnosis.valuationReady=true,v=>v.snapshot.reviewDigest='bad']) {
    const value=fixture();mutate(value);assert.throws(()=>historicalReview(value,'assignment','edition','review'));
  }
});
test('inventory rejects mixed targets and unexpected ready or retry states',()=>{
  const target={assignmentId:'assignment',parentEditionId:'edition'};
  const baseline={...target,fundamentalsDigest:'a'.repeat(64),sourcePackFingerprint:'b'.repeat(64),parentEditionDigest:'c'.repeat(64)};
  const base={valuationReady:false,truncated:false,baseline};
  function pack(){return [{...base,originals:[]},{...base,...target,retrievals:[]},{...base,...target,reviews:[]}];}
  assert.equal(historicalInventory(...pack(),'assignment','edition').reviews.reviews.length,0);
  for (const mutate of [v=>v[1].parentEditionId='other',v=>v[2].assignmentId='other',v=>v[0].valuationReady=true,v=>v[1].retrievals=[{status:'unresolved',retryAllowed:true}],v=>v[2].reviews=[{status:'accepted',requiresEvidenceRecheck:true}],v=>v[0].originals=[{id:'s',kind:'unknown'}]]) {
    const value=pack();mutate(value);assert.throws(()=>historicalInventory(...value,'assignment','edition'));
  }
});

test('parallel inventories cannot combine different source packs or financial baselines',()=>{
  const target={assignmentId:'assignment',parentEditionId:'edition'};
  const baseline={...target,fundamentalsDigest:'a'.repeat(64),sourcePackFingerprint:'b'.repeat(64),parentEditionDigest:'c'.repeat(64)};
  const pack=()=>[{baseline:{...baseline},originals:[],valuationReady:false,truncated:false},
    {...target,baseline:{...baseline},retrievals:[],valuationReady:false,truncated:false},
    {...target,baseline:{...baseline},reviews:[],valuationReady:false,truncated:false}];
  assert.doesNotThrow(()=>historicalInventory(...pack(),'assignment','edition'));
  for(const key of ['fundamentalsDigest','sourcePackFingerprint','parentEditionDigest']) {
    for(let index=0;index<3;index++) {
      const changed=pack();changed[index].baseline[key]='d'.repeat(64);
      assert.throws(()=>historicalInventory(...changed,'assignment','edition'));
      const missing=pack();delete missing[index].baseline[key];
      assert.throws(()=>historicalInventory(...missing,'assignment','edition'));
    }
  }
  const legacy=pack();delete legacy[1].baseline;
  assert.throws(()=>historicalInventory(...legacy,'assignment','edition'));
});
