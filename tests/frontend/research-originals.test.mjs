import test from 'node:test';import assert from 'node:assert/strict';import {webcrypto} from 'node:crypto';
import {verifiedOriginal} from '../../src/research-originals.mjs';
const raw=new TextEncoder().encode('Synthetic original bytes'),digest=b=>webcrypto.subtle.digest('SHA-256',b);
const hash=[...new Uint8Array(await digest(raw))].map(x=>x.toString(16).padStart(2,'0')).join('');
const source={id:'src-id',originalHash:hash};
test('original retrieval verifies exact bytes and a typed read-only route',async()=>{
 let observed;
 const blob=await verifiedOriginal('https://example.com','stock-analysis','run id',source,{digest,fetcher:async(url,options)=>{observed={url,options};return new Response(raw);}});
 assert.equal(await blob.text(),'Synthetic original bytes');assert.match(observed.url,/stock-analysis-run\/run%20id\/source\/src-id$/);assert.equal(observed.options.method,undefined);
});
test('changed bytes, unavailable originals and invalid route never become downloads',async()=>{
 await assert.rejects(verifiedOriginal('','company','id',source,{digest,fetcher:async()=>new Response('changed')}),/do not match/);
 await assert.rejects(verifiedOriginal('','company','id',source,{digest,fetcher:async()=>new Response(JSON.stringify({error:'Original changed'}),{status:409})}),/Original changed/);
 await assert.rejects(verifiedOriginal('','evil','id',source,{fetcher:()=>{throw Error('must not fetch');}}),/identity/);
});
