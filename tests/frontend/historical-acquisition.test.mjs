import test from 'node:test';
import assert from 'node:assert/strict';
import {acquisitionDetails} from '../../src/historical-acquisition.mjs';
const value={status:'recorded_unverified',verified:false,provider:'fixture',observedUrl:'https://example.test/prices',retrievedAt:'2026-10-12T12:00:00Z',collectorVersion:'v1'};
test('acquisition display distinguishes recorded claims from missing evidence',()=>{
 assert.match(acquisitionDetails(value).message,/requires verification/);
 assert.equal(acquisitionDetails(value).url,value.observedUrl);
 assert.match(acquisitionDetails().message,/not recorded/);
});
test('unsafe or falsely verified acquisition details are not displayed',()=>{
 for(const v of [{...value,verified:true},{...value,observedUrl:'https://example.test/?key=secret'},{...value,observedUrl:'javascript:alert(1)'},{...value,retrievedAt:'bad'},{status:'unavailable'}])assert.equal(acquisitionDetails(v).url,undefined);
});

import {acquisitionWarnings} from '../../src/historical-acquisition.mjs';
const selection={priceSource:{id:'p'},earningsSource:{id:'s'}};
const consistent={verified:false,status:'consistent_unverified',metadataDigest:'a'.repeat(64),gaps:[]};
test('selected acquisition warnings keep consistency separate from verification',()=>{
 const rows=acquisitionWarnings({p:consistent,s:{verified:false,status:'missing',metadataDigest:null,gaps:['missing_acquisition_metadata']}},selection);
 assert.match(rows[0],/^Price source:.*not verified/);
 assert.match(rows[1],/^Earnings source:.*not recorded/);
 assert.match(acquisitionWarnings(undefined,selection)[0],/unavailable/);
 assert.match(acquisitionWarnings({p:{...consistent,status:'mismatch',gaps:['endpoint_or_issuer_mismatch','retrieval_date_mismatch']}},selection)[0],/saved company.*retrieval date/);
});
test('wrong source, false verification and contradictory consistency checks fail closed',()=>{
 for(const checks of [{other:consistent},{p:{...consistent,verified:true}},{p:{...consistent,gaps:['provider_mismatch']}},{p:{...consistent,status:'mismatch'}},{p:{...consistent,status:'mismatch',gaps:['invented']}}])assert.throws(()=>acquisitionWarnings(checks,selection),/Acquisition checks/);
});
