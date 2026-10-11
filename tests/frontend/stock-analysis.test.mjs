import test from 'node:test';import assert from 'node:assert/strict';
import {reportStats,thesisDraft} from '../../src/stock-analysis-model.mjs';
const run={status:'complete',sources:[{id:'s'}],baseline:{body:{thesis:'Original thesis',assumptions:[{id:'retained'}],operatingModel:{original:true}}},state:{report:{summary:{investment_thesis:['Supported draft','Unmatched']},monitor:{breakers:['Unreviewed breaker']}},citations:{'/summary/investment_thesis/0':{statement:'Supported draft',evidence:[{sourceId:'s',matched:true}],review:'supported',passageMatched:true},'/summary/investment_thesis/1':{statement:'Unmatched',review:'supported',passageMatched:false},'/monitor/breakers/0':{statement:'Unreviewed breaker',review:'needs_review',passageMatched:true}}}};
test('draft uses reviewed source-matched statements and preserves analyst model and assumptions',()=>{const d=thesisDraft(run);assert.equal(d.thesis,'Supported draft');assert.equal(d.assumptions[0].id,'retained');assert.deepEqual(d.operatingModel,{original:true});assert.equal(d.changeConditions,'');assert.equal(run.baseline.body.thesis,'Original thesis');assert.equal(reportStats(run).unresolved,2);});
test('partial or unsupported research cannot become a thesis draft',()=>{assert.throws(()=>thesisDraft({...run,status:'running'}));assert.throws(()=>thesisDraft({...run,state:{citations:{}}}));});

import {readRoute,routeHash,viewGroup,viewLabel} from '../../src/workspace-model.mjs';
test('stock analysis belongs to Companies and carries company context in its route',()=>{assert.equal(viewGroup('stockanalysis').id,'companies');assert.equal(viewLabel('stockanalysis'),'Stock analysis');assert.deepEqual(readRoute(routeHash('stockanalysis','SYK')),{view:'stockanalysis',ticker:'SYK'});});

import {supported} from '../../src/stock-analysis-evidence.mjs';
import {researchCoverage} from '../../src/research-coverage-model.mjs';
test('badges preserve zero and reject stale wording, unknown sources and unavailable claims',()=>{
 const c={statement:'0',review:'supported',passageMatched:true,evidence:[{sourceId:'s',matched:true}]};
 assert.equal(supported(c,0,[{id:'s'}]),true);
 assert.equal(supported(c,1,[{id:'s'}]),false);
 assert.equal(supported(c,0,[]),false);
 assert.equal(supported({...c,statement:'Unavailable'},'Unavailable',[{id:'s'}]),false);
 assert.equal(supported({...c,passageMatched:'yes'},0,[{id:'s'}]),false);
});
test('headline stats match field coverage and orphan citations cannot enter a thesis draft',()=>{
 const altered=structuredClone(run);
 altered.state.citations['/summary/investment_thesis/9']={...altered.state.citations['/summary/investment_thesis/0'],statement:'Orphan assertion'};
 const c=researchCoverage(altered),stats=reportStats(altered);
 assert.equal(stats.supported,c.matched);assert.equal(stats.unresolved,c.review);assert.equal(stats.unavailable,c.unavailable);
 assert.equal(thesisDraft(altered).thesis,'Supported draft');
 altered.state.report.summary.investment_thesis[0]='Changed wording';
 assert.throws(()=>thesisDraft(altered));
});
