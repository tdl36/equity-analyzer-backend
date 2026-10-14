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
import {recentLabDocuments} from '../../src/pipeline-lab-model.mjs';
test('recency respects labeled dates, excludes unknown and future dates, and publisher search works',()=>{
 const rows=labDocuments([{id:'thesis_1',filename:'a.pdf',publishedDate:'2026-10-01',publisher:'Broker A'},{id:'thesis_2',filename:'b.pdf',createdAt:'2025-01-01'},{id:'thesis_3',filename:'c.pdf'},{id:'thesis_4',filename:'d.pdf',publishedDate:'2027-01-01'}],[]);
 assert.equal(recentLabDocuments(rows,'30',Date.parse('2026-10-14')).length,1);
 assert.equal(recentLabDocuments(rows,'all').length,4);
 assert.equal(visibleLabDocuments(rows,'Broker A','All','date')[0].dateLabel,'Published');
});
import {labCollectionStatus} from '../../src/pipeline-lab-model.mjs';
test('collection history never claims a live feed or successful pending delivery',()=>{
 assert.equal(labCollectionStatus(null,'ABT'),'Collection status unavailable');
 const data={updatedAt:new Date().toISOString(),snapshot:{policies:[{ticker:'ABT',lastSuccess:'2026-10-01'}],requests:[]}};
 assert.match(labCollectionStatus(data,'ABT'),/verified through 2026-10-01/);
 data.snapshot.requests=[{ticker:'ABT',status:'needs_auth'}];assert.match(labCollectionStatus(data,'ABT'),/sign in/);
 data.snapshot.requests=[{ticker:'ABT',status:'queued'}];assert.match(labCollectionStatus(data,'ABT'),/pending/);
 data.updatedAt='2020-01-01';assert.match(labCollectionStatus(data,'ABT'),/stale/);
});
import {taskSourceGuide} from '../../src/pipeline-lab-model.mjs';
test('task source guidance separates thesis challenge from event notes without scoring contents',()=>{
 assert.match(taskSourceGuide('Transcripts','thesis'),/contrary evidence/);
 assert.match(taskSourceGuide('Transcripts','note'),/this event/);
 assert.match(taskSourceGuide('Broker research','thesis'),/same underlying evidence/);
 assert.match(taskSourceGuide('Broker research','note'),/Attribute views/);
 assert.match(taskSourceGuide('unknown','thesis'),/Review the original/);
 assert.notEqual(taskSourceGuide('Filings','note'),taskSourceGuide('Filings','thesis'));
});
