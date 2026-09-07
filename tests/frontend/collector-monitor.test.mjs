import test from 'node:test';
import assert from 'node:assert/strict';
import {filterDocuments, collectionCounts} from '../../src/collector-monitor-model.mjs';
const docs = [
  {ticker:'DE',filename:'Earnings call',publisher:'Company',status:'handed_off',usage:'research'},
  {ticker:'DE',filename:'Broker original',publisher:'Analyst',status:'staged',usage:'reference_only'},
  {ticker:'ABT',filename:'Results',publisher:'Company',status:'staged',usage:'research'},
  {ticker:'AMT',filename:'Report',publisher:'Analyst',status:'duplicate',usage:'research'},
];
test('held originals are exceptions, never pending handoffs',()=>{
  assert.deepEqual(collectionCounts(docs),{saved:1,held:1,pending:1,duplicate:1});
  assert.equal(filterDocuments(docs,{status:'pending'})[0].ticker,'ABT');
});
test('company, disposition, and text filters intersect without changing source records',()=>{
  assert.deepEqual(filterDocuments(docs,{ticker:'DE',status:'held',query:' ANALYST '}),[docs[1]]);
  assert.equal(filterDocuments(docs,{ticker:'ABT',status:'held'}).length,0);
  assert.equal(filterDocuments(docs).length,4);
});
