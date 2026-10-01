import test from 'node:test';
import assert from 'node:assert/strict';
import {companySnapshot,compareSnapshots,snapshotDocument} from '../../src/company-snapshot-model.mjs';
const assumption={id:'a',claim:'Margin recovery',support:'Reported expansion',contrary:'Mix pressure',nextTest:'Next earnings',evidenceType:'interpretation'};
const version={revision:2,created_at:'2026-10-01T12:00:00Z',body:{thesis:'Recovery thesis',assumptions:[assumption],evidenceLinks:[{assumptionId:'a',field:'support',after:'Old wording',evidence:[{excerpt:'Old quote'}]},{assumptionId:'a',field:'support',after:'Reported expansion',evidence:[{excerpt:'Exact original passage',source:{filename:'Original.pdf'}}]}]}};
test('snapshot keeps saved identity and only links excerpts matching current field wording',()=>{
 const result=companySnapshot('ABC',version);
 assert.equal(result.revision,2);assert.equal(result.savedAt,version.created_at);
 assert.deepEqual(result.assumptions[0].evidence.map(e=>e.excerpt),['Exact original passage']);
 assert.ok(result.gaps.includes('Dated market expectations'));
 assert.equal(version.body.evidenceLinks.length,2);
});
test('missing baseline does not manufacture a thesis or verified coverage',()=>{
 const result=companySnapshot('ABC',null);assert.equal(result.revision,0);assert.equal(result.thesis,'');assert.ok(result.gaps.includes('Investment thesis'));
});
test('comparison uses stable assumption identity and preserves removed contrary evidence',()=>{
 const before=companySnapshot('ABC',version);
 const after=companySnapshot('ABC',{revision:3,body:{...version.body,assumptions:[{...assumption,contrary:''},{id:'b',claim:'New hypothesis'}]}});
 const diff=compareSnapshots(before,after);
 assert.ok(diff.some(c=>c.before==='Mix pressure'&&c.after===''));
 assert.ok(diff.some(c=>c.label==='Assumption added'));
 const removed=compareSnapshots(after,companySnapshot('ABC',{revision:4,body:{assumptions:[]}}));
 assert.equal(removed.filter(c=>c.label==='Assumption removed').length,2);
});
test('export preserves version, gaps and source wording and escapes active HTML',()=>{
 const result=companySnapshot('ABC',version);result.thesis='<script>alert(1)</script>';
 const html=snapshotDocument(result);
 assert.ok(html.includes('revision 2'));assert.ok(html.includes('Original.pdf'));
 assert.ok(html.includes('Exact original passage'));assert.ok(html.includes('Calibri'));
 assert.ok(html.includes('&lt;script&gt;'));assert.ok(!html.includes('<script>'));
 assert.ok(html.includes('Dated market expectations'));
});
