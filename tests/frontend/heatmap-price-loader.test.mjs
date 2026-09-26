import test from 'node:test';
import assert from 'node:assert/strict';
import {loadHeatmapPrices} from '../../src/heatmap-price-loader.mjs';
test('parallel loading is bounded and overlapping stocks are reused',async()=>{
 let active=0,peak=0,calls=0;const progress=[];
 const request=async(url,opt)=>{calls++;active++;peak=Math.max(peak,active);await new Promise(r=>setTimeout(r,5));active--;return {fetchedAt:new Date().toISOString(),quotes:Object.fromEntries(JSON.parse(opt.body).tickers.map(t=>[t,{changePct:1}]))};};
 const args={api:'cache-test',tickers:Array.from({length:101},(_,i)=>'T'+i),period:'1d',signal:new AbortController().signal,request,onUpdate:()=>{},onProgress:n=>progress.push(n)};
 assert.equal(await loadHeatmapPrices(args),0);assert.equal(calls,6);assert.equal(peak,3);assert.equal(progress.at(-1),101);
 await loadHeatmapPrices({...args,tickers:['T1','T2']});assert.equal(calls,6);
 await loadHeatmapPrices({...args,tickers:['T1'],period:'1m'});assert.equal(calls,7);
});
test('failed batches remain retryable and cancellation stops callbacks',async()=>{
 let updates=0;const controller=new AbortController();
 const args={api:'failure-test',tickers:['X'],period:'1d',signal:controller.signal,request:async()=>{throw Error('down');},onUpdate:()=>updates++,onProgress:()=>{}};
 assert.equal(await loadHeatmapPrices(args),1);assert.equal(await loadHeatmapPrices(args),1);
 await loadHeatmapPrices({...args,request:async()=>{controller.abort();return {quotes:{X:{changePct:1}},fetchedAt:new Date().toISOString()};}});
 assert.equal(updates,0);
});
