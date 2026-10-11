import test from 'node:test';import assert from 'node:assert/strict';
import {activityRows,activityAge,macConnected,collectionHealth} from '../../src/activity-model.mjs';
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
