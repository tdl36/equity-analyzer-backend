import test from 'node:test';
import assert from 'node:assert/strict';
import {readMeetingPreferences,saveMeetingPreferences,meetingJobTiming} from '../../src/meeting-preferences.mjs';
test('valid preferences persist across manual and command entry points',()=>{
 let v;const store={getItem:()=>v,setItem:(_,x)=>{v=x;}};
 assert.equal(readMeetingPreferences('one_on_one',store).format,'one_on_one');
 saveMeetingPreferences('hosted_pm','generalist',store);
 assert.deepEqual(readMeetingPreferences('conference',store),{format:'hosted_pm',audience:'generalist'});
 v='{"format":"invalid"}';assert.equal(readMeetingPreferences('conference',store).format,'conference');
});
test('job timing treats server timestamps as UTC and flags old checkpoints',()=>{
 const text=meetingJobTiming({createdAt:'2026-09-08T19:00:00',updatedAt:'2026-09-08T19:02:00'},Date.parse('2026-09-08T19:08:00Z'));
 assert.match(text,/8 min elapsed/);assert.match(text,/6 min ago/);assert.match(text,/taking longer/);
});
