import test from 'node:test';
import assert from 'node:assert/strict';
import {workflowSteps,workflowGuideHtml,syntheticOperatingModel,syntheticExpected} from '../../src/investment-workflow-guide.mjs';

test('workflow covers the ticker-to-evidence-to-model-to-revision journey with limitations',()=>{
 const html=workflowGuideHtml();
 for(const phrase of ['Open investment case','160,000','Start Deep Research','unknown provider-call','consolidated annual revenue','Unlink revenue evidence','Restore revision','remain outstanding','Calibri','zero-share']) {
  assert.ok(html.includes(phrase),phrase);
 }
 assert.equal(workflowSteps.length,10);
 assert.ok(workflowSteps.every(s=>s.action&&s.check&&s.stop));
});
test('isolated calculator fixtures cannot edit a company case and return independent assumptions',()=>{
 const first=syntheticOperatingModel(),second=syntheticOperatingModel();
 first.scenarios.base.shares='0';
 assert.equal(second.scenarios.base.shares,'100');assert.equal(first.scenarios.bear.shares,'100');
 assert.equal(first.ticker,undefined);assert.equal(first.baseRevenueObservation,undefined);
 assert.equal(syntheticExpected.impliedPrice,'21.95');
});
