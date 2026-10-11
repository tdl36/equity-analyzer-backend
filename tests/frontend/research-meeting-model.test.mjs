import test from 'node:test';
import assert from 'node:assert/strict';
import {questionSelection,originLink} from '../../src/research-meeting-model.mjs';
const pack={fingerprint:'saved',questions:[{index:0},{index:2}]};
test('meeting selections preserve indices and require valid date/type',()=>{
 assert.deepEqual(questionSelection(pack,[2,0],'2026-10-12','management'),{fingerprint:'saved',indices:[0,2],meetingDate:'2026-10-12',meetingType:'management'});
 for(const indices of [[],[1],[0,0],[true]])assert.throws(()=>questionSelection(pack,indices,'2026-10-12','management'));
 assert.throws(()=>questionSelection(pack,[0],'2026-02-30','management'));
 assert.throws(()=>questionSelection(pack,[0],'2026-10-12','other-type'));
});
test('meeting source link always points to the exact saved report',()=>{
 const origin={ticker:'UNH',runId:'6977b8d3-e286-5848-b377-159dcbfc0f21'};
 assert.equal(originLink(origin),'#view=stockanalysis&ticker=UNH&section=stock-analysis&work=analysis%3A6977b8d3-e286-5848-b377-159dcbfc0f21');
 assert.equal(originLink({...origin,ticker:'<script>'}),'');assert.equal(originLink({...origin,runId:'bad'}),'');
});
