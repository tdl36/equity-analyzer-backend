import test from 'node:test';
import assert from 'node:assert/strict';
import {eventResearch} from '../../src/earnings-model.mjs';
const event={id:'a',ticker:'MDT',activityType:'earnings_recap',input:{topic:'MDT F1Q27 Earnings'},status:'pending_review'};
test('queued recap is not represented as completed evidence',()=>{const [e]=eventResearch([event]);assert.equal(e.state,'queued');assert.equal(e.draft,false);assert.equal(e.sources.length,0);});
test('failed regeneration retains draft but still surfaces failure',()=>{const [e]=eventResearch([{...event,status:'failed',output:{synthesisMarkdown:'Previous draft',sourceFiles:['transcript.pdf'],fileCount:2}}]);assert.equal(e.state,'failed');assert.equal(e.draft,true);assert.equal(e.sourceMismatch,true);});
test('coverage names are deduplicated and unmatched names remain available',()=>{const [e]=eventResearch([{...event,output:{synthesisMarkdown:'Draft',sourceFiles:['transcript.pdf','transcript.pdf','broker.pdf',null],fileCount:2}}]);assert.equal(e.state,'draft');assert.equal(e.sources.length,2);assert.equal(e.coverage[0].files.length,1);assert.equal(e.coverage[1].files.length,0);assert.equal(e.sourceMismatch,false);});
test('only identifiable catalyst events appear and overlapping records deduplicate',()=>{assert.equal(eventResearch([event,event,{...event,id:'b',input:{}},{...event,id:'c',activityType:'news'}]).length,1);});
