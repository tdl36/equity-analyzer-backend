import test from 'node:test';
import assert from 'node:assert/strict';
import {researchCoverage,coverageDocument} from '../../src/research-coverage-model.mjs';
const claim=(statement,extra={})=>({statement,review:'supported',passageMatched:true,evidence:[{sourceId:'s',matched:true}],...extra});
const fixture=()=>({id:'id',ticker:'TEST',status:'complete',sources:[{id:'s',filename:'Original',sourceUrl:'https://example.com',coverage:{mode:'selected',omittedCharacters:123,limitation:'Selected portions'},textExtraction:{limitations:['OCR uncertainty']}}],state:{report:{summary:{one_liner:'Operating claim',core_question:'Unreviewed question',investment_thesis:[]},valuation:{multiple:'Unavailable in the source pack'}},citations:{'/summary/one_liner':claim('Operating claim'),'/summary/core_question':claim('Unreviewed question',{review:'needs_review',reviewReason:'Qualification missing'})}}});
test('coverage separates matched evidence, unresolved fields and unavailable items',()=>{
 const r=fixture(),before=structuredClone(r),c=researchCoverage(r);
 assert.equal(c.matched,1);assert.equal(c.review,1);assert.equal(c.rows.find(x=>x.id==='summary').unavailable,1);
 assert.equal(c.rows.find(x=>x.id==='valuation').review,0);assert.ok(c.noEvidence.includes('Valuation'));
 assert.equal(c.limitedSources,1);assert.equal(c.unknownCoverage,0);assert.equal(c.sources[0].omitted,123);
 assert.deepEqual(r,before);
});
test('orphan citations, changed wording and missing original references cannot count as matched',()=>{
 for(const change of [{statement:'Different wording'},{evidence:[{sourceId:'unknown',matched:true}]},{passageMatched:false},{evidence:[]}]) {
  const r=fixture();Object.assign(r.state.citations['/summary/one_liner'],change);r.state.citations['/summary/orphan']=claim('Absent from report');
  assert.equal(researchCoverage(r).matched,0);assert.equal(researchCoverage(r).review,2);
 }
});
test('missing coverage metadata never implies full-document review',()=>{
 const r=fixture();delete r.sources[0].coverage;delete r.sources[0].textExtraction;
 r.baselineStale=true;const c=researchCoverage(r);assert.equal(c.unknownCoverage,1);assert.equal(c.baselineStale,true);
 assert.match(coverageDocument(r),/coverage metadata unavailable/);
});
test('export escapes saved text and rejects executable source URLs',()=>{
 const r=fixture();r.ticker='<script>alert(1)</script>';r.sources[0].sourceUrl='javascript:alert(1)';r.state.report.summary.core_question='<img src=x onerror=alert(1)>';
 const h=coverageDocument(r);assert.ok(!h.includes('<script>'));assert.ok(!h.includes('<img'));assert.ok(!h.includes('javascript:'));assert.match(h,/&lt;img/);assert.match(h,/Calibri/);
});
