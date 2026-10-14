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
