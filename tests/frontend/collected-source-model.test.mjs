import test from 'node:test';
import assert from 'node:assert/strict';
import {collectedSources,duplicateSelections} from '../../src/collected-source-model.mjs';
import {visibleLabDocuments} from '../../src/pipeline-lab-model.mjs';
const job={id:'collection',ticker:'ABT',result:{sources:[{filename:'recent-release.pdf',sha256:'a'.repeat(64)},{filename:'old-transcript.pdf',sha256:'b'.repeat(64)},{filename:'unknown.pdf',sha256:'c'.repeat(64)}]}};
const sample=()=>({collectionId:job.id,ticker:'ABT',documents:job.result.sources.map((r,i)=>({id:'thesis_'+(i+1),filename:r.filename,receiptSha256:r.sha256,availability:'available',publishedDate:['2026-10-01','2023-07-01',null][i],publisher:'Company',sourceKind:i===0?'press-release':''}))});
test('collected sources retain receipt identity and separate unknown publication dates from upload times',()=>{
 const d=sample();d.documents.reverse();d.documents[0].createdAt='2026-10-15';const rows=collectedSources(d,job);
 assert.deepEqual(visibleLabDocuments(rows,'','All','date').map(r=>r.filename),job.result.sources.map(r=>r.filename));
 assert.equal(rows.find(r=>r.filename==='unknown.pdf').date,null);
 assert.equal(rows.find(r=>r.filename==='recent-release.pdf').typeRecorded,true);
 assert.equal(visibleLabDocuments(rows,'company','Transcripts','date').length,1);
});
test('missing and restricted originals stay visible; duplicate receipt bytes block a second selection without deleting either',()=>{
 const d=sample();d.documents[1].availability='missing';d.documents[1].id=null;d.documents[2].availability='restricted';
 assert.equal(collectedSources(d,job).length,3);
 const rows=[{filename:'a',receiptSha256:'same'},{filename:'b',receiptSha256:'same'},{filename:'c',receiptSha256:'other'}];
 assert.deepEqual([...duplicateSelections(rows,['a'])],['b']);assert.equal(rows.length,3);
});
test('mismatched collection, company, receipts or identities cannot replace the source inventory',()=>{
 for(const change of [d=>d.collectionId='other',d=>d.ticker='CVS',d=>d.documents.pop(),d=>d.documents[0].filename='other.pdf',d=>d.documents[0].receiptSha256='d'.repeat(64),d=>d.documents[0].id='bad',d=>d.documents[0].availability='unknown',d=>d.documents[0].publisher={},d=>d.documents[0].id=d.documents[1].id]){
 const d=sample();change(d);assert.throws(()=>collectedSources(d,job));
 }
});
