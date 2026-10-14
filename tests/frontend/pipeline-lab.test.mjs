import test from 'node:test';
import assert from 'node:assert/strict';
import {labDocuments,visibleLabDocuments} from '../../src/pipeline-lab-model.mjs';
import {readRoute,viewLabel} from '../../src/workspace-model.mjs';
test('Lab preserves original route and has a distinct route',()=>{
 assert.equal(readRoute('#view=pipeline').view,'pipeline');
 assert.equal(readRoute('#view=pipelinelab').view,'pipelinelab');
 assert.equal(viewLabel('pipelinelab'),'Research Pipeline Lab');
});
test('same-name documents remain distinct across storage and folders',()=>{
 const rows=labDocuments([{id:'thesis_1',filename:'Call transcript.pdf',createdAt:'2026-01-01'}],[{filename:'Call transcript.pdf',folder:'main',modified:'2026-02-01'},{filename:'Call transcript.pdf',folder:'Catalysts',modified:'2026-03-01'}]);
 assert.equal(rows.length,3);assert.equal(new Set(rows.map(r=>r.key)).size,3);
 assert.ok(rows.every(r=>r.repeated));
 assert.equal(visibleLabDocuments(rows,'','Transcripts','date')[0].folder,'Catalysts');
 assert.equal(visibleLabDocuments(rows,'Catalysts','All','date').length,1);
 assert.throws(()=>labDocuments([{filename:'x'}],[]));
 assert.throws(()=>labDocuments([], [{filename:'x'},{filename:'x'}]));
});
