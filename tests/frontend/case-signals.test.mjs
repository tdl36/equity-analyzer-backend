import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ageDays,retention,pressure,falsification,variant,divergence} from '../../src/case-signals-model.mjs';
const now=new Date('2026-09-12T12:00:00Z');
test('half-life decays relevance, never resolves an active challenge',()=>{
 const r={assumptionId:'a',asOf:'2026-08-13',kind:'event',direction:'challenge',status:'active',targetType:'risk'};
 assert.equal(retention(r,undefined,now),.5);
 const [p]=pressure([{id:'a'}],[r],undefined,now);assert.equal(p.challenge,.5);assert.equal(p.unresolved,1);
 assert.equal(retention({...r,kind:'structural'},undefined,now),1);
 assert.equal(retention({...r,status:'retired'},undefined,now),0);
 assert.equal(ageDays('2026-02-30',now),null);assert.equal(ageDays('2026-09-13',now),null);
});
test('threshold directions and stale/missing observations remain distinct',()=>{
 const r={current:'10',threshold:'12',operator:'lte',asOf:'2026-09-12',unit:'%',period:'FY26',source:'original'};
 assert.equal(falsification(r,now).distance,-2);
 assert.equal(falsification({...r,operator:'gte'},now).distance,2);
 assert.match(falsification({...r,asOf:'2025-01-01'},now).status,/older/);
 assert.equal(falsification({...r,current:''},now).distance,null);
});
test('variant gaps require dated baseline and handle zero without infinite upside',()=>{
 const r={view:'2',market:'0',asOf:'2026-09-12',source:'broker',unit:'USD',period:'FY27',baselineType:'broker'};
 assert.equal(variant(r,now).delta,2);assert.equal(variant(r,now).percent,null);
 assert.equal(variant({...r,baselineType:'unknown'},now).delta,null);
});
test('divergence asks questions and does not infer holdings or active weight',()=>{
 const p={portfolio:'Test',weight:'1',benchmarkWeight:'2',benchmark:'Index',asOf:'2026-09-12',conviction:'high'};
 const result=divergence(p,[],now);assert.equal(result.active,-1);assert.match(result.questions[0],/mandate/);
 assert.equal(divergence({...p,benchmark:''},[],now).active,null);
 assert.equal(divergence({...p,asOf:'2025-01-01'},[],now).active,null);
 assert.match(divergence({},[],now).questions[0],/Record/);
});
