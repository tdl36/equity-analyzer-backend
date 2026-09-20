import test from 'node:test';
import assert from 'node:assert/strict';
import {labDocument,documentHtml,emailDocument,labFanoutPlan,looksLikeHtmlDocument,youtubeLanguagePayload} from '../../src/summary-lab-format.mjs';
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
