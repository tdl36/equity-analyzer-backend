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

// Render the real JSX, rather than mirroring the preview-selection expression.
test('model workspace renders empty, newly added and removed drafts', async()=>{
 const {build}=await import('esbuild');
 const {createRequire}=await import('node:module');
 const require=createRequire(import.meta.url);
 const React=require('react'),{renderToStaticMarkup}=require('react-dom/server');
 const bundle=await build({entryPoints:['src/operating-model.jsx'],bundle:true,write:false,platform:'node',format:'cjs',external:['react']});
 const module={exports:{}};new Function('require','module','exports',bundle.outputFiles[0].text)(require,module,module.exports);
 const {OperatingModel,newOperatingModel}=module.exports;
 const render=(body,dirty)=>renderToStaticMarkup(React.createElement(OperatingModel,{body,dirty,ticker:'SYNTH',revision:0,busy:false,onChange:()=>{},onSave:()=>{}}));
 assert.match(render({},false),/Add EV\/EBITDA model/);
 assert.match(render({operatingModel:newOperatingModel()},true),/Calculate draft/);
 assert.match(render({},true),/Add EV\/EBITDA model/);
});

test('source receipt export preserves provenance and explicitly limits verification',async()=>{
 const {revenueEvidenceHtml}=await import('../../src/operating-model-view.mjs');
 const html=revenueEvidenceHtml({token:'1.0',unit:'billions',currency:'USD',valueMillions:'1000',fiscalYear:2025,filename:'<original>',excerpt:'<script>not executable</script>',researchRunId:'run',originalHash:'original-hash',extractionHash:'text-hash',locator:'p.2',basis:'GAAP'});
 for(const value of ['analyst','not independently verified','historical','original-hash','text-hash','&lt;script&gt;','1.0 billions'])assert.ok(html.includes(value),value);
 assert.ok(!html.includes('<script>'));
});

test('historical EBITDA comparison and evidence survive saved exports',()=>{
 const m=structuredClone(model);
 Object.assign(m,{baseEbitda:'200',baseMarginPct:'20.00',baseEbitdaReference:'Synthetic baseline',baseEbitdaObservation:{token:'200',unit:'millions',currency:'USD',valueMillions:'200',fiscalYear:2025,basis:'Adjusted EBITDA',excerpt:'Synthetic EBITDA',originalHash:'ebitda-hash'}});
 for(const row of Object.values(m.results))Object.assign(row,{marginChangePp:'0.00',ebitdaGrowthPct:'21.00'});
 const html=operatingModelHtml(m);
 for(const value of ['Base EBITDA evidence','ebitda-hash','Historical EBITDA: 200','20.00%','21.00%','percentage points','Synthetic baseline'])assert.ok(html.includes(value),value);
 assert.ok(!operatingModelHtml(model).includes('Historical EBITDA:'));
});

test('EBITDA reconciliation export keeps signed adjustments and escapes manual references',()=>{
 const m=structuredClone(model);
 m.ebitdaReconciliation={startingEbitda:'180',startingBasis:'Reported EBITDA',sourceReference:'<source>',totalAdjustments:'20',reconciledEbitda:'200',adjustments:[{label:'Gain reversal',amount:'-10',recurrence:'uncertain',reference:'<script>source</script>'}]};
 const html=operatingModelHtml(m);
 for(const text of ['Reported-to-adjusted','Gain reversal: -10','uncertain','&lt;source&gt;','&lt;script&gt;','Reconciled base EBITDA: 200','analyst judgments'])assert.ok(html.includes(text),text);
 assert.ok(!html.includes('<script>'));
 assert.ok(!operatingModelHtml(model).includes('Reported-to-adjusted'));
});

test('historical balance sheet export remains separate from target debt',()=>{
 const m={...model,baseCash:'0',baseDebt:'300',baseCashBasis:'Cash equivalents',baseDebtBasis:'Including leases',baseCashReference:'Synthetic',baseDebtReference:'<original>',historicalNetDebt:'300.00'};
 const html=operatingModelHtml(m);
 for(const value of ['Year-end Cash: 0','Year-end Debt: 300','Including leases','Historical net debt: 300.00','Target-period scenario debt remains separate','&lt;original&gt;'])assert.ok(html.includes(value),value);
});
