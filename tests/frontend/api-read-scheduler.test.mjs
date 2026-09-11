import test from 'node:test';
import assert from 'node:assert/strict';
import {createReadScheduler} from '../../src/api-read-scheduler.mjs';
const tick=()=>new Promise(resolve=>setImmediate(resolve));
test('API reads respect the concurrency bound and all queued work runs',async()=>{
 const schedule=createReadScheduler(2);let active=0,max=0;const releases=[];
 const jobs=Array.from({length:5},(_,i)=>schedule(()=>new Promise(resolve=>{active++;max=Math.max(max,active);releases.push(()=>{active--;resolve(i);});})));
 await tick();assert.equal(releases.length,2);
 for(let i=0;i<5;i++){releases[i]();await tick();}
 assert.deepEqual(await Promise.all(jobs),[0,1,2,3,4]);assert.equal(max,2);
});
test('aborted queued reads never reach the network',async()=>{
 const schedule=createReadScheduler(1);let release,called=false;const first=schedule(()=>new Promise(resolve=>{release=resolve;}));
 const controller=new AbortController();const second=schedule(()=>{called=true;},controller.signal);const rejected=assert.rejects(second,{name:'AbortError'});controller.abort();await rejected;await tick();release();await first;await tick();assert.equal(called,false);
});
test('failed reads release capacity and are not automatically repeated',async()=>{
 const schedule=createReadScheduler(1);let calls=0;
 const first=schedule(()=>{calls++;throw Error('Unavailable');});
 const second=schedule(()=>42);await assert.rejects(first,/Unavailable/);assert.equal(await second,42);assert.equal(calls,1);
});
