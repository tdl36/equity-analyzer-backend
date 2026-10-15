import test from 'node:test';
import assert from 'node:assert/strict';
import {emptyWorkDraft} from '../../src/research-workbench-recovery.mjs';
import {latestWorkForComparison,workDifferences} from '../../src/work-comparison.mjs';
const id='11111111-1111-4111-8111-111111111111',record={id,revision:2};
const row=()=>({id,revision:3,body:{...emptyWorkDraft('followup'),caseRevision:4,title:'Newer saved work'}});
test('comparison validates exact work, kind and non-regressing revision without mutating baseline',()=>{
 const r=row(),before=structuredClone(r);const saved=latestWorkForComparison({records:[r]},record,'ABT','followup');
 assert.equal(saved.revision,3);assert.equal(record.revision,2);assert.deepEqual(r,before);
 for(const records of [[],[r,r],[{...r,revision:1}],[{...r,body:{...r.body,kind:'model'}}],[{...r,body:{...r.body,caseRevision:null}}],[{...r,body:{...r.body,inputs:{}}}]])assert.throws(()=>latestWorkForComparison({records},record,'ABT','followup'));
 assert.throws(()=>latestWorkForComparison({},record,'ABT','followup'));
});
test('comparison includes changed evidence hashes, numeric zero, nested inputs and condition identity',()=>{
 const draft=emptyWorkDraft('model'),saved=structuredClone(draft);
 saved.sourceHash='b'.repeat(64);saved.inputs.baselineEPS=0;saved.reviewConditions=[{id,category:'valuation',trigger:'Observe',state:'unassessed',evidence:'',sourceReference:''}];
 const rows=workDifferences(draft,saved);assert.deepEqual(rows.map(r=>r.key),['sourceHash','inputs.baselineEPS','reviewConditions']);
 assert.equal(rows[1].saved,'0');assert.equal(rows[1].local,'Not recorded');assert.match(rows[2].saved,new RegExp(id));
 assert.deepEqual(workDifferences(draft,structuredClone(draft)),[]);
});
test('server snapshots and calculations are not copied into the editable comparison',()=>{
 const r=row();r.body.calculation={value:99};r.body.answerSnapshot={response_notes:'Snapshot'};
 const saved=latestWorkForComparison({records:[r]},record,'ABT','followup');
 assert.equal(saved.draft.calculation,undefined);assert.equal(saved.draft.answerSnapshot,undefined);
});
