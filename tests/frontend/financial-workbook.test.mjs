import test from 'node:test';
import assert from 'node:assert/strict';
import {financialWorkbook} from '../../src/financial-workbook-download.mjs';
const id='6977b8d3-e286-5848-b377-159dcbfc0f21';
test('workbook download uses saved assignment and rejects error or non-workbook responses',async()=>{
 let path;const blob=await financialWorkbook('',id,{fetcher:async p=>{path=p;return {ok:true,arrayBuffer:async()=>new Uint8Array([80,75,3,4]).buffer};}});
 assert.equal(path,`/api/research/assignments/${id}/financial-history`);assert.equal(blob.size,4);
 await assert.rejects(()=>financialWorkbook('',id,{fetcher:async()=>({ok:false,json:async()=>({error:'No saved annual financial observations'})})}),/No saved/);
 await assert.rejects(()=>financialWorkbook('',id,{fetcher:async()=>({ok:true,arrayBuffer:async()=>new TextEncoder().encode('<html>error</html>').buffer})}),/verified/);
 await assert.rejects(()=>financialWorkbook('','bad'),/saved research/);
});
