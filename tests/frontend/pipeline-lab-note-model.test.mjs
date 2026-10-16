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
