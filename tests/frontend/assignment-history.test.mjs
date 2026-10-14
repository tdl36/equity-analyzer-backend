import test from 'node:test';import assert from 'node:assert/strict';
import {historyPage,appendHistory} from '../../src/assignment-history-model.mjs';
const row={id:'00000000-0000-4000-8000-000000000001',ticker:'ABT',status:'complete'};
test('archive accepts bounded pages and rejects foreign ticker or invalid destinations',()=>{
 const page={readOnly:true,assignments:[row],nextCursor:'cursor'};
 assert.equal(historyPage(page,'ABT').rows.length,1);
 for(const value of [{...page,readOnly:false},{...page,assignments:[{...row,id:'bad'}]},{...page,assignments:[row,row]},{...page,nextCursor:'x'.repeat(513)}])assert.throws(()=>historyPage(value));
 assert.throws(()=>historyPage(page,'UNH'));
 assert.deepEqual(historyPage({...page,assignments:[],nextCursor:null}).rows,[]);
});
test('pagination preserves loaded rows and de-duplicates overlapping pages',()=>{
 const old=[row],other={...row,id:'00000000-0000-4000-8000-000000000002'};
 assert.deepEqual(appendHistory(old,[row,other]),[row,other]);assert.deepEqual(old,[row]);
});
