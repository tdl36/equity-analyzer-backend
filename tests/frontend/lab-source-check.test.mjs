import test from 'node:test';
import assert from 'node:assert/strict';
import {checkedLabSourceCheck} from '../../src/lab-source-check.mjs';
const ref='charlie:thesis_1', row={reference:ref,filename:'a.pdf',bytes:12,sha256:'a'.repeat(64)};
const result={task:'thesis',revision:1,checkedOnly:true,generationAvailable:true,verified:[row],blockers:[],digest:'b'.repeat(64)};
test('source preflight requires complete saved selection before generation controls',()=>{
 assert.equal(checkedLabSourceCheck(result,'thesis',1,[ref]),result);
 for(const change of [{task:'note'},{revision:2},{checkedOnly:false},{verified:[]},{verified:[null]},{verified:[{...row,bytes:0}]},{verified:[{...row,sha256:'bad'}]},{digest:null},{blockers:['missing']}])assert.throws(()=>checkedLabSourceCheck({...result,...change},'thesis',1,[ref]));
 assert.throws(()=>checkedLabSourceCheck(result,'thesis',1,[ref,'charlie:thesis_2']));
 assert.throws(()=>checkedLabSourceCheck(result,'thesis',1,['charlie:thesis_3']));
});
test('blocked and older-server source checks remain readable without enabling generation',()=>{
 const blocked={...result,verified:[],blockers:['Import required'],digest:null,generationAvailable:false};
 assert.equal(checkedLabSourceCheck(blocked,'thesis',1,['icloud:pending']),blocked);
 assert.equal(checkedLabSourceCheck({...result,generationAvailable:undefined},'thesis',1,[ref]).generationAvailable,undefined);
});
