import test from 'node:test';
import assert from 'node:assert/strict';
import {operatingModelHtml} from '../../src/operating-model-view.mjs';
import {companySnapshot,compareSnapshots,snapshotDocument} from '../../src/company-snapshot-model.mjs';
const model={version:'ev-ebitda-v1',currency:'USD',baseYear:2025,targetYear:2027,baseRevenue:'1000',referencePrice:'20',asOf:'2026-10-01',revenueReference:'<script>synthetic</script>',priceReference:'Synthetic price',ebitdaBasis:'Adjusted',warnings:['Manual references unverified'],scenarios:{},results:{}};
for(const n of ['bear','base','bull']){model.scenarios[n]={growthPct:'10',marginPct:'20',multiple:'10',netDebt:'200',otherClaims:'50',nonOperatingAssets:'25',shares:'100',rationale:'Synthetic scenario'};model.results[n]={revenue:'1210.00',ebitda:'242.00',enterpriseValue:'2420.00',equityValue:'2195.00',impliedPrice:'21.95',priceReturnPct:'9.75'};}
test('saved-case exports preserve matching scenario assumptions, formula, revision and uncertainty',()=>{
 const snapshot=companySnapshot('SYNTH',{revision:7,body:{operatingModel:model}}),html=snapshotDocument(snapshot);
 for(const expected of ['revision 7','ev-ebitda-v1','FY2027','21.95','9.75%','Target net debt','net debt − other claims','unverified'])assert.ok(html.includes(expected),expected);
 assert.ok(html.includes('&lt;script&gt;'));assert.ok(!html.includes('<script>'));
 assert.equal(operatingModelHtml(null),'');
});
test('model edits and removal appear in historical case comparisons',()=>{
 const before=companySnapshot('SYNTH',{revision:7,body:{operatingModel:model}});
 const changed=structuredClone(model);changed.scenarios.base.growthPct='11';
 const after=companySnapshot('SYNTH',{revision:8,body:{operatingModel:changed}});
 assert.equal(compareSnapshots(before,after).filter(x=>x.label.includes('Operating model')).length,1);
 assert.ok(compareSnapshots(before,companySnapshot('SYNTH',{revision:9,body:{}})).some(x=>x.after==='No model'));
});
