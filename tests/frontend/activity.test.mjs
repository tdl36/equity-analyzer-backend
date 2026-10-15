import test from 'node:test';import assert from 'node:assert/strict';
import {activityDestination,activityRows,activityAge,macConnected,collectionHealth,workLane,nextAction,workOverview,focusRows,recentAssignments} from '../../src/activity-model.mjs';
import {readRoute,routeHash,VIEWS} from '../../src/workspace-model.mjs';
test('activity is reachable without company context',()=>{assert.ok(VIEWS.includes('activity'));assert.deepEqual(readRoute(routeHash('activity')),{view:'activity',ticker:''});});
test('filters find work across tickers without mixing attention and active',()=>{const data={items:[{ticker:'UNH',title:'Research',step:'Review',bucket:'active'},{ticker:'SYK',title:'Collection',step:'Sign in',bucket:'attention'}]};assert.equal(activityRows(data,'active','unh').length,1);assert.equal(activityRows(data,'attention','SYK').length,1);assert.equal(activityRows(data,'active','SYK').length,0);});
test('unknown/stale Mac status never presents as connected',()=>{const now=Date.parse('2026-10-10T12:00:00Z');assert.equal(macConnected(null,now),false);assert.equal(macConnected('2026-10-10 11:59:00',now),true);assert.equal(macConnected('2026-10-10T11:00:00Z',now),false);assert.equal(activityAge('2026-10-10 10:45:00',now),'1h 15m ago');assert.equal(activityAge(null,now),'Time unavailable');});
test('delivery health separates heartbeat, sync and full command queue',()=>{
 const now=Date.parse('2026-10-10T12:00:00Z'),data={asOf:'2026-10-10T12:00:00Z',agentLastSeen:'2026-10-10T11:59:00Z',collectionLastSeen:'2026-10-10T11:00:00Z',collectionDelivery:{pending:1,oldestPendingAt:'2026-10-10T11:40:00Z'}};
 assert.deepEqual(collectionHealth(data,now).map(x=>x.ok),[true,false,false]);
 assert.deepEqual(collectionHealth({...data,collectionLastSeen:data.asOf,collectionDelivery:{pending:0}},now).map(x=>x.ok),[true,true,true]);
 assert.equal(collectionHealth({...data,collectionDelivery:null},now)[2].value,'Unknown');
 assert.equal(collectionHealth({...data,unavailable:['Collection delivery']},now)[2].value,'Unknown');
 assert.deepEqual(collectionHealth({...data,asOf:'2026-10-10T11:00:00Z'},now).map(x=>x.ok),[false,false,false]);
 assert.equal(macConnected('2026-10-11T12:00:00Z',now),false);
 assert.equal(collectionHealth({...data,collectionDelivery:{pending:1,oldestPendingAt:'2026-10-10T12:01:00Z'}},now)[2].ok,false);
});

test('exception inbox distinguishes decisions, waiting and reported processing',()=>{
 const rows=[
 {id:'w',bucket:'active',status:'waiting',stale:true,workflow:'auto_evidence_intake'},
 {id:'p',bucket:'active',status:'processing',stale:false},
 {id:'b',bucket:'active',status:'running',stale:true},
 {id:'r',bucket:'attention',status:'awaiting_approval'},
 {id:'o',bucket:'older',status:'running',stale:true},
 {id:'d',bucket:'recent',status:'complete'}];
 assert.deepEqual(rows.map(workLane),['waiting','processing','blocked','review','older','recent']);
 assert.deepEqual(workOverview({items:rows}),{waiting:1,processing:1,blocked:1,review:1,older:1,recent:1});
 assert.deepEqual(focusRows({items:rows}).map(r=>r.id),['b','r','p','w']);
 assert.equal(activityRows({items:rows},'focus').length,2);
 assert.equal(activityRows({items:rows},'waiting').length,1);
 assert.match(nextAction(rows[0]),/daily limits/);
 assert.match(nextAction(rows[2]),/may still be running/);
 assert.match(nextAction(rows[3]),/will not approve/);
 assert.match(nextAction({status:'needs_auth',bucket:'attention'}),/directly in the source app/);
});

test('Activity carries the exact supported job identity without inventing links',()=>{
 const jobId='00000000-0000-4000-8000-000000000123';
 assert.equal(activityDestination({workflow:'research_assignment',jobId}).work,'assignment:'+jobId);
 assert.equal(activityDestination({workflow:'stock_analysis_runs',jobId}).work,'analysis:'+jobId);
 assert.equal(activityDestination({id:'collection:'+jobId,jobId}).work,'collection:'+jobId);
 assert.equal(activityDestination({workflow:'research_assignment',jobId:'bad'}).work,undefined);
});

test('due research issue opens its exact saved review and requests an investor follow-up',()=>{
 const row={workflow:'research_decision',jobId:'00000000-0000-4000-8000-000000000001',bucket:'attention',status:'needs_review'};
 assert.equal(workLane(row),'review');assert.match(nextAction(row),/follow-up review/);
 assert.deepEqual(activityDestination(row),{view:'stockanalysis',section:'reviews',work:'decision:'+row.jobId});
});

test('uncertain legacy execution is surfaced for reconciliation without retry advice',()=>{const row={workflow:'agent_runs',status:'running',bucket:'attention',executionUnconfirmed:true,ticker:'WELL'};assert.equal(workLane(row),'blocked');assert.equal(focusRows({items:[row]}).length,1);assert.match(nextAction(row),/Do not submit another paid run/);});

test('unstarted analyst jobs explain start recovery without a paid retry',()=>{
 const row={workflow:'agent_runs',status:'queued',bucket:'active',pendingStart:true,startRecovery:true};
 assert.equal(workLane(row),'waiting');assert.match(nextAction(row),/first execution/);
 assert.match(nextAction({...row,bucket:'attention'}),/backend worker availability/);
 assert.match(nextAction({...row,startRecovery:false}),/unavailable/);
 assert.doesNotMatch(nextAction(row),/collection delivery/i);
});

test('Home retains recent completed assignment links without treating collection as output',()=>{
 const ids=Array.from({length:5},(_,i)=>`00000000-0000-4000-8000-00000000000${i}`);
 const row=(i)=>({jobId:ids[i],workflow:'research_assignment',ticker:'ABT',bucket:'recent',status:'complete',updatedAt:`2026-10-10T12:0${i}:00Z`});
 const items=[...ids.map((_,i)=>row(i)),row(4),{...row(3),workflow:'collection_control'},{...row(2),status:'cancelled'},{...row(1),jobId:'bad'},{...row(0),bucket:'attention'}];
 const before=JSON.stringify(items);
 assert.deepEqual(recentAssignments({items}).map(r=>r.jobId),[ids[4],ids[3],ids[2]]);
 assert.equal(activityDestination(recentAssignments({items})[0]).work,'assignment:'+ids[4]);
 assert.equal(JSON.stringify(items),before);
 assert.deepEqual(recentAssignments({items:[{...row(0),bucket:'older'}]}),[]);
});

test('evidence review retains exact proposal identity',()=>{
 const id='11111111-1111-4111-8111-111111111111';
 assert.deepEqual(activityDestination({workflow:'evidence_amendment',jobId:id}),{view:'desk',section:'evidence',work:'amendment:'+id});
});
