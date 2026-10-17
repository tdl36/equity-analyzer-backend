import test from 'node:test';
import assert from 'node:assert/strict';
import {rewriteScope,rewriteStillMatches} from '../../src/package-rewrite-model.mjs';
const fields=[{id:'thesis',label:'Thesis',text:'Reasoned view',limit:40000},{id:'brief',label:'Brief',text:'Short view',limit:1600}];
test('single section freezes rest of package as read-only context',()=>{
 const body=rewriteScope(fields,'brief');assert.deepEqual(body.fields,[fields[1]]);assert.deepEqual(body.contextFields,[fields[0]]);
 assert.equal(rewriteStillMatches(body,fields),true);
 assert.equal(rewriteStillMatches(body,[{...fields[0],text:'Changed view'},fields[1]]),false);
 assert.equal(rewriteStillMatches(body,[...fields,{id:'new'}]),false);
});
test('whole package freezes all fields and legacy requests still compare only their selected fields',()=>{
 assert.deepEqual(rewriteScope(fields,'*'),{fields,contextFields:[]});
 assert.equal(rewriteStillMatches({fields:[fields[1]]},[{...fields[0],text:'Changed'},fields[1]]),true);
 assert.equal(rewriteStillMatches({fields:[fields[1]]},[fields[0]]),false);
});
