import test from 'node:test';
import assert from 'node:assert/strict';
import {labDocuments,visibleLabDocuments} from '../../src/pipeline-lab-model.mjs';
import {readRoute,routeHash,viewLabel} from '../../src/workspace-model.mjs';
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

test('recorded collection kind takes precedence over filename hints without inventing an author',()=>{
 const rows=labDocuments([{id:'thesis_99',filename:'presentation.pdf',sourceKind:'broker-report',author:'Observed analyst'}],[]);
 assert.equal(rows[0].type,'Broker research');assert.equal(rows[0].typeRecorded,true);
});

test('Lab output route retains the exact note identity across reloads',()=>{
 const id='11111111-1111-4111-8111-111111111111';
 assert.deepEqual(readRoute(routeHash('pipelinelab','ABT','','labnote:'+id)),{view:'pipelinelab',ticker:'ABT',work:'labnote:'+id});
 assert.equal(readRoute('#view=pipelinelab&ticker=ABT&work=labnote:bad').work,undefined);
 assert.equal(readRoute(routeHash('pipeline','ABT','','labnote:'+id)).work,undefined);
});


test('publication-only review excludes recent uploads and preserves the inventory',()=>{
 const now=Date.parse('2026-10-14');
 const rows=labDocuments([
  {id:'thesis_1',filename:'recent.pdf',publishedDate:'2026-10-01'},
  {id:'thesis_2',filename:'old.pdf',publishedDate:'2025-01-01',createdAt:'2026-10-13'},
  {id:'thesis_3',filename:'upload.pdf',createdAt:'2026-10-13'},
  {id:'thesis_4',filename:'invalid.pdf',publishedDate:'invalid'},
  {id:'thesis_5',filename:'future.pdf',publishedDate:'2027-01-01'}
 ],[{filename:'modified.pdf',modified:'2026-10-13'}]);
 assert.deepEqual(recentLabDocuments(rows,'30',now,true).map(r=>r.filename),['recent.pdf']);
 assert.deepEqual(recentLabDocuments(rows,'all',now,true).map(r=>r.filename),['recent.pdf','old.pdf']);
 assert.equal(recentLabDocuments(rows,'all',now).length,6);
 assert.equal(rows.length,6);
});

import {groupLabDocuments} from '../../src/pipeline-lab-model.mjs';
test('source grouping preserves identities and ordering without changing selections',()=>{
 const rows=[{key:'a',type:'Transcripts'},{key:'b',type:'Broker research'},{key:'c',type:'Transcripts'}];
 assert.deepEqual(groupLabDocuments(rows).map(g=>[g.label,g.rows.map(r=>r.key)]),[['Transcripts',['a','c']],['Broker research',['b']]]);
 assert.deepEqual(groupLabDocuments(rows,false)[0].rows,rows);
 assert.deepEqual(rows.map(r=>r.key),['a','b','c']);
 assert.deepEqual(groupLabDocuments([]),[]);
});

import {sameLabSelection} from '../../src/pipeline-lab-model.mjs';
test('restoring the same selected identities clears dirty state regardless of order',()=>{
 assert.equal(sameLabSelection({thesis:['b','a'],note:[]},{thesis:['a','b'],note:[]}),true);
 assert.equal(sameLabSelection({thesis:['b'],note:['a']},{thesis:['a','b'],note:[]}),false);
});

import {labSourceDate,labSourceTime} from '../../src/pipeline-lab-model.mjs';
test('publication calendar dates do not shift back a day in New York',()=>{
 for(const value of ['2026-01-01','2026-07-01','2026-11-01','2024-02-29']){
  const [year,month,day]=value.split('-').map(Number);
  assert.equal(labSourceDate(value),`${month}/${day}/${year}`);
 }
 assert.equal(labSourceDate('2026-10-15T01:00:00Z'),'10/14/2026');
 assert.equal(labSourceTime('2026-10-15 01:00:00'),Date.parse('2026-10-15T01:00:00Z'));
 assert.equal(labSourceDate('2026-10-15 01:00:00'),'10/14/2026');
 for(const value of ['2026-02-30','2025-02-29','invalid',null]){
  assert.equal(labSourceDate(value),'Unknown');assert.ok(Number.isNaN(labSourceTime(value)));
 }
 const rows=labDocuments([{id:'thesis_1',filename:'bad.pdf',publishedDate:'2026-02-30'}],[]);
 assert.equal(recentLabDocuments(rows,'all',Date.parse('2026-03-04'),true).length,0);
});
test('unknown source kinds remain suggested rather than recorded',()=>{
 for(const sourceKind of ['future-kind','__proto__','constructor','']){
  const row=labDocuments([{id:'thesis_1',filename:'transcript.pdf',sourceKind}],[])[0];
  assert.equal(row.type,'Transcripts');assert.equal(row.typeRecorded,false);
 }
});
