import test from 'node:test';import assert from 'node:assert/strict';
import {researchAlerts} from '../../src/research-alert-model.mjs';
test('alerts existing failures immediately, groups parent and child, not running or complete',()=>{
 const items=[{id:'c',ticker:'ABT',workflow:'stock_analysis_runs',status:'attention',error:'bad source'},{id:'p',ticker:'ABT',workflow:'research_assignment',status:'attention',error:'bad source'},{id:'r',ticker:'X',workflow:'stock_analysis_runs',status:'running'},{id:'d',ticker:'Y',workflow:'stock_analysis_runs',status:'complete'}];
 const rows=researchAlerts({items});assert.equal(rows.length,1);assert.equal(rows[0].id,'p');
});
test('dismissal does not suppress a new failure on the same job',()=>{
 const item={id:'p',ticker:'ABT',workflow:'research_assignment',status:'attention',error:'first'};
 const key=researchAlerts({items:[item]})[0].alertKey;
 assert.equal(researchAlerts({items:[item]},{[key]:true}).length,0);
 assert.equal(researchAlerts({items:[{...item,error:'second'}]},{[key]:true}).length,1);
});

import {incompleteResearchAlerts} from '../../src/research-alert-model.mjs';
test('warns when blocked research coverage is capped or unavailable',()=>{
 assert.equal(incompleteResearchAlerts({truncated:['Research workflows (attention)']}),true);
 assert.equal(incompleteResearchAlerts({truncated:['Research workflows']}),true);
 assert.equal(incompleteResearchAlerts({unavailable:['Stock analysis']}),true);
 assert.equal(incompleteResearchAlerts({truncated:['Research workflows (active)','Audio processing (attention)']}),false);
});
