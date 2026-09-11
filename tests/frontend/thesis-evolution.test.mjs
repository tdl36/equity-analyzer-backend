import test from 'node:test';
import assert from 'node:assert/strict';
import {compareCases,pillarState,latestWork} from '../../src/thesis-evolution-model.mjs';
test('stable identities distinguish revised, removed and new pillars',()=>{
 const a={thesis:'Old',assumptions:[{id:'a',claim:'Margin',support:'10%'},{id:'b',claim:'Cash'}]};
 const b={thesis:'New',assumptions:[{id:'a',claim:'Margin',support:'11%'},{id:'c',claim:'Growth'}]};
 assert.equal(pillarState(a,b,'a'),'Revised');assert.equal(pillarState(a,b,'b'),'Removed');assert.equal(pillarState(a,b,'c'),'Added');
 assert.equal(compareCases(a,b).length,4);assert.equal(compareCases(b,b).length,0);
 assert.equal(pillarState(null,b,'a',false),'Baseline');assert.equal(pillarState(null,b,'b',false),'—');
});
test('work reconstruction selects versions only from supplied history',()=>{
 const rows=[{id:'a',revision:1},{id:'b',revision:2},{id:'a',revision:3}];
 assert.equal(latestWork(rows).find(r=>r.id==='a').revision,3);
 assert.equal(latestWork(rows.slice(0,2)).find(r=>r.id==='a').revision,1);
});
