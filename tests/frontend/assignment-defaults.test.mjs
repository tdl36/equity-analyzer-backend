import test from 'node:test';import assert from 'node:assert/strict';
import {checkedAssignmentDefaults,outputNames} from '../../src/assignment-defaults.mjs';
test('saved output defaults validate without leaking mutable references',()=>{
 const v={outputs:['visual','thesis'],horizon:'12 months'};const d=checkedAssignmentDefaults(v);v.outputs.push('meeting_prep');assert.deepEqual(d.outputs,['visual','thesis']);assert.equal(outputNames(d.outputs),'One-pager, Thesis proposal');
});
test('invalid saved defaults cannot configure an assignment',()=>{
 for(const v of [null,{}, {outputs:[],horizon:'x'},{outputs:['visual','visual'],horizon:'x'},{outputs:['other'],horizon:'x'},{outputs:['visual'],horizon:' '},{outputs:['visual'],horizon:'x'.repeat(101)}])assert.throws(()=>checkedAssignmentDefaults(v));
});
