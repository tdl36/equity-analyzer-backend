import test from 'node:test';
import assert from 'node:assert/strict';
import {readEditBuffer,writeEditBuffer} from '../../src/pipeline-lab-edit-buffer.mjs';
test('session note buffers retain exact base revision, title and prose',()=>{
 const values=new Map(),store={getItem:k=>values.get(k)??null,setItem:(k,v)=>values.set(k,v)};
 assert.equal(readEditBuffer(store,'a'),null);
 const value={revision:7,title:'Draft',markdown:'Original\n\nNew thinking'};
 writeEditBuffer(store,'a',value);assert.deepEqual(readEditBuffer(store,'a'),value);
 assert.equal(readEditBuffer(store,'b'),null);
 assert.throws(()=>writeEditBuffer(store,'a',{...value,revision:0}));
 assert.deepEqual(readEditBuffer(store,'a'),value);
});
test('corrupt, unbounded or silently dropped buffers never report success',()=>{
 for(const raw of ['{','null','[]',JSON.stringify({revision:1,title:'x',markdown:3}),JSON.stringify({revision:1,title:'x',markdown:'x',accepted:true}),'x'.repeat(2100001)])assert.throws(()=>readEditBuffer({getItem:()=>raw},'key'));
 assert.throws(()=>writeEditBuffer({getItem:()=>null,setItem:()=>{}},'key',{revision:1,title:'',markdown:''}));
});
