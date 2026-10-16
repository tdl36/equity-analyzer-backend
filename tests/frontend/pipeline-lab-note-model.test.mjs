import test from 'node:test';import assert from 'node:assert/strict';
import {checkedNote,checkedNoteHistory} from '../../src/pipeline-lab-note-model.mjs';
const note={noteId:'one',ticker:'ABT',revision:2,title:'Saved',markdown:'Exact text',qualityIssues:[],sourcePack:[],acceptance:null};
test('exact note reads preserve content and explicit or unknown acceptance',()=>{
 for(const acceptance of [undefined,null,{revision:2,acceptedAt:'2026-10-15T10:00:00Z'}]){
  const value={...note,acceptance};assert.equal(checkedNote(value,'one','ABT',2),value);
  assert.equal(checkedNote(value,'one','ABT'),value);
 }
});
test('different note company revision or malformed response never substitutes a saved note',()=>{
 for(const patch of [{noteId:'two'},{ticker:'CVS'},{revision:3},{revision:0},{revision:'2'},{markdown:null},{title:3},{qualityIssues:null},{sourcePack:null}])
  assert.throws(()=>checkedNote({...note,...patch},'one','ABT',2));
 assert.throws(()=>checkedNote(null,'one','ABT'));
});
test('acceptance must refer to the exact saved revision with a recorded decision time',()=>{
 for(const acceptance of [{revision:1,acceptedAt:'2026-10-15'},true,{}, {revision:2,acceptedAt:'invalid'}])
  assert.throws(()=>checkedNote({...note,acceptance},'one','ABT'));
});
const rows=[3,2].map(revision=>({revision,ticker:'ABT',title:'Version',createdAt:'2026-10-15T10:00:00Z'}));
test('history validates exact note descending revisions and continuation independently',()=>{
 const value={noteId:'one',versions:rows,nextBefore:2};assert.equal(checkedNoteHistory(value,'one','ABT'),value);
 const older={noteId:'one',versions:[{...rows[0],revision:1}],nextBefore:null};assert.equal(checkedNoteHistory(older,'one','ABT',2),older);
 assert.deepEqual(checkedNoteHistory({noteId:'one',versions:[],nextBefore:null},'one','ABT').versions,[]);
});
test('wrong note company duplicate or out-of-page history cannot replace metadata',()=>{
 const value={noteId:'one',versions:rows,nextBefore:2};
 for(const patch of [{noteId:'two'},{versions:null},{versions:[rows[0],rows[0]]},{versions:[{...rows[0],ticker:'CVS'}]},{nextBefore:3},{nextBefore:0},{nextBefore:undefined},{versions:[],nextBefore:2}])
  assert.throws(()=>checkedNoteHistory({...value,...patch},'one','ABT'));
 assert.throws(()=>checkedNoteHistory(value,'one','ABT',3));
});

const {confirmedNoteEdit}=await import('../../src/pipeline-lab-note-model.mjs');
const body={eventId:'stable-edit',expectedRevision:1,title:'  Edited  ',markdown:'Frozen edit text'};
const receipt={noteId:'one',revision:2,status:'draft',replayed:false};
const edited={...note,title:'Edited',markdown:body.markdown};
test('edit reconciliation reads the receipt revision even when newer edits exist',async()=>{
 for(const replayed of [false,true]){
  const calls=[];
  const result=await confirmedNoteEdit({id:'one',ticker:'ABT',body,
   post:async(path,value)=>{calls.push(path);assert.equal(value,body);return {...receipt,replayed};},
   read:async path=>{calls.push(path);return edited;}});
  assert.equal(result,edited);
  assert.deepEqual(calls,['/api/pipeline-lab/notes/one/revisions','/api/pipeline-lab/notes/one?revision=2']);
 }
});
test('invalid edit receipts cannot clear recovery or request another saved note',async()=>{
 for(const patch of [{noteId:'other'},{revision:3},{revision:'2'},{status:'accepted'},{replayed:undefined}]){
  await assert.rejects(confirmedNoteEdit({id:'one',ticker:'ABT',body,
   post:async()=>({...receipt,...patch}),read:async()=>assert.fail('No read after invalid receipt')}),/receipt/);
 }
});
test('saved edit text and exact revision are verified before confirmation',async()=>{
 for(const patch of [{markdown:'Someone else’s edit'},{title:'Different'},{revision:3},{ticker:'CVS'},{noteId:'other'}]){
  await assert.rejects(confirmedNoteEdit({id:'one',ticker:'ABT',body,
   post:async()=>receipt,read:async()=>({...edited,...patch})}));
 }
});
test('only definitive write rejection releases an edit, never subsequent read failure',async()=>{
 for(const status of [400,409,500,undefined]){
  const error=Object.assign(new Error('Synthetic failure'),{status});
  await assert.rejects(confirmedNoteEdit({id:'one',ticker:'ABT',body,
   post:async()=>{throw error;},read:async()=>assert.fail('No read after rejected write')}),
   e=>e.editRejected===[400,409].includes(status));
  await assert.rejects(confirmedNoteEdit({id:'one',ticker:'ABT',body,
   post:async()=>receipt,read:async()=>{throw error;}}),e=>!e.editRejected);
 }
});
