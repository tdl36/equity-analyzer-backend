import test from 'node:test';
import assert from 'node:assert/strict';
import {checkedMeetingContext as check} from '../../src/meeting-context-model.mjs';
const answer={id:1,meeting_id:24,question:'Question?',response_notes:null,status:'planned'};
const version={id:8,meeting_id:24,version:1,topics:[{topic:'Topic',questions:[{question:'Question?'}]}]};
const quality={meetingId:24,questionSetId:8,questions:1,highPriority:1,withVerifiedPassages:0,citedSources:0,availableSources:0,duplicateQuestions:0,scope:'Synthetic'};
test('independent meeting reads validate exact scope',()=>{assert.equal(check('quality',quality,24,8),quality);assert.doesNotThrow(()=>check('answers',{meetingId:24,answers:[answer]},24,8));assert.doesNotThrow(()=>check('versions',{meetingId:24,versions:[version],truncated:true},24,8));});
test('cross-meeting and wrong-version responses are rejected',()=>{assert.throws(()=>check('quality',quality,24,9));assert.throws(()=>check('quality',{...quality,meetingId:25},24,8));assert.throws(()=>check('answers',{meetingId:24,answers:[{...answer,meeting_id:25}]},24,8));assert.throws(()=>check('versions',{meetingId:24,versions:[{...version,meeting_id:25}],truncated:false},24,8));});
test('duplicate and malformed histories fail without replacing loaded state',()=>{for(const versions of [[version,version],[{...version,topics:{}}],[{...version,topics:[{topic:'Bad',questions:[{}]}]}]])assert.throws(()=>check('versions',{meetingId:24,versions,truncated:false},24,8));assert.throws(()=>check('answers',{meetingId:24,answers:[answer,answer]},24,8));assert.throws(()=>check('quality',{...quality,questions:-1},24,8));});
