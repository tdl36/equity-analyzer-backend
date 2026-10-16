import test from 'node:test';import assert from 'node:assert/strict';
import {meetingAgendaState} from '../../src/activity-model.mjs';
const item={id:24,ticker:'ABT',date:'2026-10-15',prepared:true,type:'management'};
const agenda={from:'2026-10-15',through:'2026-10-21',truncated:false,items:[item]};
test('shared Home/Activity agenda distinguishes verified empty from missing/failed reads',()=>{
 assert.deepEqual(meetingAgendaState({meetingAgenda:agenda}),agenda);
 assert.deepEqual(meetingAgendaState({meetingAgenda:{...agenda,items:[]}}).items,[]);
 for(const data of [null,{}, {meetingAgenda:null},{meetingAgenda:agenda,unavailable:['Meeting preparation agenda']}])assert.equal(meetingAgendaState(data),null);
});
test('agenda rejects malformed, duplicate and out-of-window links instead of inventing empty coverage',()=>{
 for(const change of [{id:0},{id:'24'},{id:NaN},{ticker:''},{date:'2026-10-22'},{date:'2026-10-14'},{date:'2026-02-30'},{prepared:'false'},{type:{}}])assert.equal(meetingAgendaState({meetingAgenda:{...agenda,items:[{...item,...change}]}}),null);
 for(const change of [{from:'bad'},{through:'2026-10-22'},{truncated:null},{items:[item,item]},{items:Array.from({length:51},(_,i)=>({...item,id:i+1}))}])assert.equal(meetingAgendaState({meetingAgenda:{...agenda,...change}}),null);
 assert.equal(meetingAgendaState({meetingAgenda:{...agenda,truncated:true}}).truncated,true);
});
