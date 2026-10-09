import test from 'node:test';import assert from 'node:assert/strict';
import {reportStats,thesisDraft} from '../../src/stock-analysis-model.mjs';
const run={status:'complete',baseline:{body:{thesis:'Original thesis',assumptions:[{id:'retained'}],operatingModel:{original:true}}},state:{citations:{'/summary/investment_thesis/0':{statement:'Supported draft',review:'supported',passageMatched:true},'/summary/investment_thesis/1':{statement:'Unmatched',review:'supported',passageMatched:false},'/monitor/breakers/0':{statement:'Unreviewed breaker',review:'needs_review',passageMatched:true}}}};
test('draft uses reviewed source-matched statements and preserves analyst model and assumptions',()=>{const d=thesisDraft(run);assert.equal(d.thesis,'Supported draft');assert.equal(d.assumptions[0].id,'retained');assert.deepEqual(d.operatingModel,{original:true});assert.equal(d.changeConditions,'');assert.equal(run.baseline.body.thesis,'Original thesis');assert.equal(reportStats(run).unresolved,2);});
test('partial or unsupported research cannot become a thesis draft',()=>{assert.throws(()=>thesisDraft({...run,status:'running'}));assert.throws(()=>thesisDraft({...run,state:{citations:{}}}));});

import {readRoute,routeHash,viewGroup,viewLabel} from '../../src/workspace-model.mjs';
test('stock analysis belongs to Companies and carries company context in its route',()=>{assert.equal(viewGroup('stockanalysis').id,'companies');assert.equal(viewLabel('stockanalysis'),'Stock analysis');assert.deepEqual(readRoute(routeHash('stockanalysis','SYK')),{view:'stockanalysis',ticker:'SYK'});});
