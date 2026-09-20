import test from 'node:test';
import assert from 'node:assert/strict';
import {sourceMarkers,labDocument,documentHtml,emailDocument,labFanoutPlan,labProgressSummary,looksLikeHtmlDocument,youtubeLanguagePayload} from '../../src/summary-lab-format.mjs';
test('renders source labels, headings and lists without executing source HTML',()=>{
 const html=labDocument('# Topic\n\n**Management:** uncertain [P1]\n\n- Detail\n- Caveat\n\n<script>alert(1)</script>');
 assert.match(html,/<h3>Topic<\/h3>/);assert.match(html,/<strong>Management:<\/strong>/);
 assert.match(html,/<ul><li>Detail<\/li><li>Caveat<\/li><\/ul>/);
 assert.match(html,/\[P1\]/);assert(!html.includes('<script>'));
});
test('email includes all selected sections and escapes title',()=>{
 const html=emailDocument('<img src=x>',[['Brief','First'],['Assessment','Uncertain']]);
 assert.match(html,/font-size:11pt/);assert.match(html,/Calibri/);
 assert.match(html,/Brief/);assert.match(html,/Assessment/);assert(!html.includes('<img'));
});
test('uses the page sanitizer for legacy HTML and does not show raw tags',()=>{
 const source='<h2>Topic</h2><p><strong>Result:</strong> Revenue grew.</p><script>bad()</script>';
 assert.equal(looksLikeHtmlDocument(source),true);
 const rendered=documentHtml(source,value=>value.replace(/<script[\s\S]*?<\/script>/gi,''));
 assert.match(rendered,/<h2>Topic<\/h2>/);assert.match(rendered,/<strong>Result:<\/strong>/);assert(!rendered.includes('<script>'));
});
test('HTML without a sanitizer fails closed as visible text',()=>{
 const rendered=documentHtml('<p>Result</p>');
 assert.match(rendered,/&lt;p&gt;Result&lt;\/p&gt;/);assert(!rendered.includes('<p>Result</p>'));
});
test('YouTube language options mirror Summary and never allow Korean-only without Korean',()=>{
 assert.deepEqual(youtubeLanguagePayload(false,true),{generateKorean:false,koreanOnly:false,outputMode:'english'});
 assert.deepEqual(youtubeLanguagePayload(true,false),{generateKorean:true,koreanOnly:false,outputMode:'korean_bilingual'});
 assert.deepEqual(youtubeLanguagePayload(true,true),{generateKorean:true,koreanOnly:true,outputMode:'korean_only'});
});
test('a completed transcription adopts its automatic experiment instead of running a second',()=>{
 assert.deepEqual(labFanoutPlan({status:'complete',summaryId:'s1',summaryLabId:'lab-1'}),{adoptId:'lab-1',start:false,error:'',summaryId:'s1'});
 assert.deepEqual(labFanoutPlan({status:'complete',summaryId:'s1'}),{adoptId:'',start:true,error:'',summaryId:'s1'});
});
test('a transcription without a saved Summary reports the failure and starts nothing',()=>{
 const plan=labFanoutPlan({status:'complete'});
 assert.equal(plan.start,false);assert.equal(plan.adoptId,'');assert.match(plan.error,/no saved Summary/);
});
test('progress reports verified sections, not just source parts',()=>{
 // The real MCK experiment: all five drafts on screen, four verified.
 const row={status:'running',state:{parts:{0:{},1:{},2:{}},totalParts:3,
  sections:{brief:'x',takeaways:'x',meeting:'x',questions:'x',assessment:'x'},
  completedSections:['brief','takeaways','meeting','questions']}};
 const {detail,blocked}=labProgressSummary(row,5);
 assert.match(detail,/3 of 3 source parts reviewed/);
 assert.match(detail,/4 of 5 sections verified/);
 assert.match(detail,/verifying drafts against the source/);
 assert.match(blocked,/Email all and Copy all unlock/);
});
test('the final cross-section review is named rather than looking idle',()=>{
 const row={status:'running',state:{parts:{0:{},1:{}},totalParts:2,completedSections:['a','b','c','d','e']}};
 assert.match(labProgressSummary(row,5).detail,/final cross-section review/);
});
test('a finished experiment states what was verified and blocks nothing',()=>{
 const row={status:'complete',state:{parts:{0:{},1:{},2:{}},totalParts:3,completedSections:['a','b','c','d','e']}};
 const {detail,blocked}=labProgressSummary(row,5);
 assert.equal(detail,'5 sections verified against 3 source parts');
 assert.equal(blocked,'');
});
test('an experiment still reading the source says so',()=>{
 const row={status:'running',state:{parts:{0:{}},totalParts:4,completedSections:[]}};
 assert.match(labProgressSummary(row,5).detail,/1 of 4 source parts reviewed .* reading the source/);
});
test('a korean-only experiment counts its single section',()=>{
 const row={status:'complete',state:{parts:{0:{}},totalParts:1,completedSections:['korean']}};
 assert.equal(labProgressSummary(row,1).detail,'1 section verified against 1 source part');
});

test('source citations render as one switchable superscript without touching markup or exports', () => {
  const html = labDocument('NAPD grew 5% [P1]. Oncology led [P1][P2][P1]. Policy is unclear [P3].');
  const marked = sourceMarkers(html);
  assert.equal((marked.match(/<sup class="src"/g) || []).length, 3);
  assert.ok(marked.includes('>1,2</sup>'), 'a run of citations collapses to one marker, deduplicated');
  assert.ok(marked.includes('title="Source parts 1, 2"'));
  assert.ok(marked.includes('title="Source part 3"'));
  assert.ok(!marked.includes('[P'), 'no bracketed citation survives in the reader');
  assert.ok(marked.includes('5%<sup'), 'the marker sits tight against the word it cites');
  assert.ok(marked.includes('</sup>.'), 'punctuation still follows the citation directly');
  // The stored text is what Copy, Download and Email use; it must be unchanged.
  assert.ok(html.includes('[P1]') && html.includes('[P3]'));
});

test('markers inside tags are left alone so attributes cannot be corrupted', () => {
  const input = '<a href="/x?q=%5BP1%5D" title="see [P1] note">cited [P2]</a>';
  const marked = sourceMarkers(input);
  assert.ok(marked.includes('title="see [P1] note"'), 'attribute text is not rewritten');
  assert.ok(marked.includes('href="/x?q=%5BP1%5D"'));
  assert.equal((marked.match(/<sup class="src"/g) || []).length, 1);
});

test('text without citations is returned unchanged', () => {
  assert.equal(sourceMarkers('<p>Plain note.</p>'), '<p>Plain note.</p>');
  assert.equal(sourceMarkers(''), '');
  assert.equal(sourceMarkers(), '');
});
