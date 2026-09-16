import test from 'node:test';
import assert from 'node:assert/strict';
import {labDocument,emailDocument} from '../../src/summary-lab-format.mjs';
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
