import test from 'node:test';
import assert from 'node:assert/strict';
import {managedMeetingState} from '../../src/meeting-command-state.mjs';
test('managed or unknown active status never invites a duplicate start',()=>{
  for (const job of [{status:'checking'},{status:'unavailable'},{prep_status:'queued'},{prep_status:'running',prep_step:'generating'},{prep_status:'failed'}]) assert.equal(managedMeetingState(job).blocked,true);
  assert.equal(managedMeetingState(null).blocked,false);
  assert.equal(managedMeetingState({prep_status:'done'}).blocked,false);
});
test('document progress stays distinct from question generation',()=>{
  assert.match(managedMeetingState({prep_status:'running',prep_step:'analyzing',completed:'11',total:'12'}).message,/11\/12/);
  assert.match(managedMeetingState({prep_status:'running',prep_step:'generating',completed:'12',total:'12'}).message,/Generating/);
});
