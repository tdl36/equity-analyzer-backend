import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {originalBlob,originalFilename} from '../../src/original-download.mjs';
const data='Synthetic original',hash=createHash('sha256').update(data).digest('hex');
test('download preserves exact bytes and enforces the frozen receipt when available',async()=>{
 const blob=await originalBlob(new Response(data),hash);assert.equal(await blob.text(),data);
 assert.equal(await (await originalBlob(new Response(data))).text(),data);
 await assert.rejects(()=>originalBlob(new Response('changed'),hash),/differ/);
 await assert.rejects(()=>originalBlob(new Response(data),'bad'),/receipt is invalid/);
});
test('failed, empty and oversized responses cannot be downloaded as originals',async()=>{
 await assert.rejects(()=>originalBlob(new Response(JSON.stringify({error:'Not available'}),{status:409})),/Not available/);
 await assert.rejects(()=>originalBlob(new Response('')),/empty/);
 await assert.rejects(()=>originalBlob({ok:true,blob:async()=>({size:15000001})}),/15 MB/);
 assert.equal(originalFilename('../report?.pdf'),'.._report_.pdf');
});
