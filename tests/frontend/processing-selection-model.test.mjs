import test from 'node:test';
import assert from 'node:assert/strict';
import {processingSelection} from '../../src/processing-selection-model.mjs';
const hash='a'.repeat(64);
const base=()=>({selected:['current.pdf'],supplements:['2'],since:'2026-07-01',until:'2026-10-14',collection:{loaded:true,rows:[{filename:'current.pdf',publishedDate:'2026-10-01',availability:'available',receiptSha256:hash},{filename:'excluded.pdf',availability:'restricted'}]},library:{loaded:true,cursor:'2',rows:[{id:'thesis_2',filename:'old.pdf',publishedDate:'2024-03-01',eligible:true,recordedSha256:hash},{id:'thesis_3',filename:'restricted.pdf',eligible:false}]}});
test('combined selections retain roles and show cross-library duplicates without mutating choices',()=>{
 const input=base(),copy=structuredClone(input),s=processingSelection(input);
 assert.equal(s.total,2);assert.equal(s.collection,1);assert.equal(s.supplemental,1);
 assert.deepEqual(s.dates,{known:2,unknown:0,earliest:'2024-03-01',latest:'2026-10-01',windowKnown:true,before:1,within:1,after:0});
 assert.deepEqual(s.duplicates.map(g=>g.map(r=>r.filename)),[['current.pdf','old.pdf']]);
 assert.equal(s.unselectedCollection,1);assert.equal(s.unselectedLibrary,1);assert.equal(s.libraryComplete,false);
 assert.equal(s.metadataPending,false);assert.equal(s.missing,0);assert.equal(s.unavailable,0);assert.deepEqual(input,copy);
});
test('unloaded restored selections remain explicit and prevent a complete metadata preview',()=>{
 const input=base();input.library={loaded:false,rows:[]};const s=processingSelection(input);
 assert.equal(s.total,2);assert.equal(s.missing,1);assert.equal(s.metadataPending,true);assert.equal(s.rows[1].selection,'2');assert.equal(s.rows[1].filename,'Stored original #2');assert.equal(s.dates.unknown,1);assert.equal(s.hashUnknown,1);
 input.supplements=[];assert.equal(processingSelection(input).metadataPending,false);
});
test('missing, restricted and failed reads cannot be represented as ready even when metadata is retained',()=>{
 const input=base();input.collection.rows[0].availability='missing';input.library.rows[0].eligible=false;input.library.error='offline';const s=processingSelection(input);
 assert.equal(s.unavailable,2);assert.equal(s.metadataPending,true);assert.equal(s.rows.length,2);assert.equal(s.libraryComplete,false);
 input.library.error='';input.library.busy=true;assert.equal(processingSelection(input).metadataPending,true);
 input.library.busy=false;input.collection.loaded=false;assert.equal(processingSelection(input).metadataPending,true);
});
test('publication unknowns never use upload timestamps and invalid windows are not zero coverage',()=>{
 const input=base();input.collection.rows[0].publishedDate='2026-02-30';input.collection.rows[0].createdAt='2026-10-01';input.library.rows[0].publishedDate=null;input.since='2026-12-01';const s=processingSelection(input);
 assert.deepEqual(s.dates,{known:0,unknown:2,earliest:null,latest:null,windowKnown:false,before:null,within:null,after:null});
 input.library.rows[0].publishedDate='2026-10-15';input.since='2026-07-01';assert.equal(processingSelection(input).dates.after,1);
});
test('absent or malformed hashes never prove distinct or duplicate originals',()=>{
 const input=base();input.collection.rows[0].receiptSha256='bad';input.library.rows[0].recordedSha256='bad';let s=processingSelection(input);assert.equal(s.duplicates.length,0);assert.equal(s.hashUnknown,2);
 input.collection.rows[0].receiptSha256=hash.toUpperCase();input.library.rows[0].recordedSha256=hash;s=processingSelection(input);assert.equal(s.duplicates.length,1);assert.equal(s.hashUnknown,0);
});
