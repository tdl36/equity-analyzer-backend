import test from 'node:test';
import assert from 'node:assert/strict';
import {readRoute,routeHash} from '../../src/workspace-model.mjs';
import {activityDestination} from '../../src/activity-model.mjs';
test('research sections survive direct-link serialization with ticker',()=>{
 for(const [view,section] of [['desk','evidence'],['desk','collection'],['stockanalysis','changes'],['stockanalysis','committee']])
  assert.deepEqual(readRoute(routeHash(view,'BRK.B',section)),{view,ticker:'BRK.B',section});
});
test('sections are bounded to their workspace and unknown values are discarded',()=>{
 assert.deepEqual(readRoute('#view=desk&section=changes'),{view:'desk',ticker:''});
 assert.equal(routeHash('today','DE','evidence'),'#view=today&ticker=DE');
 assert.deepEqual(readRoute('#view=stockanalysis&section=%3Cscript%3E'),{view:'stockanalysis',ticker:''});
});
test('activity opens the relevant review, collection, or research surface',()=>{
 for(const [workflow,view,section] of [['auto_evidence_intake','desk','evidence'],['evidence_amendment','desk','evidence'],['catalyst_signal','desk','collection'],['company_research_runs','stockanalysis','research'],['investment_committee_runs','stockanalysis','committee'],['research_assignment','stockanalysis','stock-analysis']])
  assert.deepEqual(activityDestination({workflow,view:'portfolio'}),{view,section});
 assert.deepEqual(activityDestination({id:'collection:123',view:'desk'}),{view:'desk',section:'collection'});
 assert.deepEqual(activityDestination({workflow:'unknown',view:'summary'}),{view:'summary',section:''});
});
test('saved meeting links survive reload and reject invalid or misplaced ids',()=>{
 assert.deepEqual(readRoute(routeHash('meetingprep','UNH','','meeting:42')),{view:'meetingprep',ticker:'UNH',work:'meeting:42'});
 for(const value of ['meeting:0','meeting:-1','meeting:01','meeting:2147483648','meeting:1:2','meeting:<script>'])assert.equal(readRoute('#view=meetingprep&work='+encodeURIComponent(value)).work,undefined);
 assert.equal(readRoute('#view=desk&work=meeting%3A42').work,undefined);
});
