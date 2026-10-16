import test from 'node:test';
import assert from 'node:assert/strict';
import {processingLibraryPage} from '../../src/processing-library-model.mjs';
const scope={ticker:'ABT',collectionId:'collection',after:null};
const document=id=>({id:'thesis_'+id,filename:'original-'+id+'.pdf',eligible:true});
const page=()=>({...scope,documents:[document('42'),document('20')],nextCursor:null});
test('library metadata binds company, collection and strict descending page identity',()=>{
 const p=page(),before=JSON.stringify(p);assert.equal(processingLibraryPage(p,scope).documents,p.documents);assert.equal(JSON.stringify(p),before);
 assert.deepEqual(processingLibraryPage({...scope,documents:[],nextCursor:null},scope),{documents:[],nextCursor:null});
 const rows=Array.from({length:200},(_,i)=>document(String(300-i)));
 assert.equal(processingLibraryPage({...scope,documents:rows,nextCursor:'101'},scope).nextCursor,'101');
 assert.equal(processingLibraryPage({...page(),after:'100'}, {...scope,after:'100'}).documents.length,2);
 const large=[document('9223372036854775807'),document('9223372036854775806')];
 assert.equal(processingLibraryPage({...page(),documents:large},scope).documents.length,2);
});
test('foreign, incomplete, repeated and out-of-order library pages cannot become selectable',()=>{
 for(const edit of [p=>p.ticker='CVS',p=>p.collectionId='other',p=>delete p.after,p=>p.after='42',p=>delete p.nextCursor,p=>p.documents=null,p=>p.documents.push(document('42')),p=>p.documents.reverse(),p=>p.documents[0].id='thesis_01',p=>p.documents[0].id='thesis_9223372036854775808',p=>delete p.documents[0].filename,p=>p.documents[0].eligible='true',p=>p.nextCursor='20',p=>p.documents=Array(201).fill(document('42'))]){
  const p=page();edit(p);assert.throws(()=>processingLibraryPage(p,scope),/could not be verified/);
 }
 assert.throws(()=>processingLibraryPage({...page(),after:'20'},{...scope,after:'20'}),/could not be verified/);
 assert.throws(()=>processingLibraryPage({...page(),after:'bad'},{...scope,after:'bad'}),/could not be verified/);
});
