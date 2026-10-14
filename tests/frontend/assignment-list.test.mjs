import test from 'node:test';
import assert from 'node:assert/strict';
import {assignmentRows} from '../../src/assignment-list.mjs';
const row={id:'f9e363e4-597e-4394-b9d4-36d83def935f',ticker:'ABT',status:'running',input:{since:'2026-01-01',until:'2026-03-01'},result:{artifacts:[]}};
test('assignment status preserves valid empty and active queues',()=>{
 assert.deepEqual(assignmentRows({assignments:[]},'ABT'),[]);
 assert.deepEqual(assignmentRows({assignments:[row]},'ABT'),[row]);
});
test('invalid or cross-company status never becomes an empty queue',()=>{
 for(const assignments of [null,{},[null],[row,row],[{...row,ticker:'UNH'}],[{...row,status:'unknown'}],[{...row,input:null}],[{...row,result:[]}],[{...row,result:{artifacts:{}}}]])assert.throws(()=>assignmentRows({assignments},'ABT'),/could not be verified/);
});
