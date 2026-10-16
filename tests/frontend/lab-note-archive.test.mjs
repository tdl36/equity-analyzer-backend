import test from 'node:test';import assert from 'node:assert/strict';
import {noteArchivePage,appendNoteArchive} from '../../src/lab-note-archive-model.mjs';
const row={noteId:'11111111-1111-4111-8111-111111111111',ticker:'ABT',title:'Saved note',status:'attention',reason:'provider_outcome_unknown',revision:2,acceptedRevision:1,createdAt:'2026-01-01T00:00:00Z'};
const page={ticker:'ABT',jobs:[row],nextCursor:'cursor',readOnly:true};
test('note archive retains exact older acceptance and unconfirmed execution status',()=>{
 assert.deepEqual(noteArchivePage(page,'ABT'),{rows:[row],cursor:'cursor'});
 assert.deepEqual(appendNoteArchive([row],[row,{...row,noteId:'22222222-2222-4222-8222-222222222222'}]).map(r=>r.noteId),[row.noteId,'22222222-2222-4222-8222-222222222222']);
 assert.deepEqual(noteArchivePage({...page,jobs:[],nextCursor:null},'ABT'),{rows:[],cursor:null});
});
test('wrong company identity, duplicates and malformed pagination never replace archive',()=>{
 for(const change of [{ticker:'UNH'},{noteId:'bad'},{revision:0},{acceptedRevision:'1'},{status:'accepted'},{createdAt:'bad'}])assert.throws(()=>noteArchivePage({...page,jobs:[{...row,...change}]},'ABT'));
 for(const change of [{ticker:'UNH'},{jobs:[row,row]},{nextCursor:7},{nextCursor:''},{readOnly:false},{jobs:null}])assert.throws(()=>noteArchivePage({...page,...change},'ABT'));
});
