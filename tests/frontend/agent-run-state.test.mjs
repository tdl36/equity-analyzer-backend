import test from 'node:test';
import assert from 'node:assert/strict';
import {agentRunPending,agentRunTerminal,agentRunHeading,mergeAgentSnapshot} from '../../src/agent-run-state.mjs';
test('queued stays pending and cannot be mistaken for completion',()=>{
 for(const status of ['queued','running']){assert.equal(agentRunPending(status),true);assert.equal(agentRunTerminal(status),false);}
 assert.equal(agentRunHeading('queued'),'Waiting to start');
 assert.equal(agentRunHeading('error'),'Run stopped');
 assert.equal(agentRunTerminal(undefined),false);
});
test('queued to running to complete retains logs and exact run identity',()=>{
 let row={runId:'a',status:'queued',logs:[]};
 for(const [i,status] of ['queued','running','complete'].entries())row=mergeAgentSnapshot(row,'a',{status,newLogs:[{message:status}],totalLogs:i+1});
 assert.equal(row.logs.length,3);assert.equal(row.status,'complete');assert.equal(agentRunTerminal(row.status),true);
 assert.equal(mergeAgentSnapshot(row,'b',{status:'error',newLogs:[],totalLogs:0}),row);
 for(const invalid of [null,{}, {status:'running',newLogs:[],totalLogs:-1}])assert.equal(mergeAgentSnapshot(row,'a',invalid),row);
});

test('actual screen poll continues through queued and failed reads, then stops on completion',async()=>{
 const {readFileSync}=await import('node:fs');
 const source=readFileSync(new URL('../../src/app.jsx',import.meta.url),'utf8');
 const prefix='const startAgentRun = ';
 const body=source.slice(source.indexOf(prefix)+prefix.length,source.indexOf('const saveAgentRun =')).trim().replace(/;$/,'');
 let row=null,tick,history=0;const clears=[],responses=[
  {ok:true,json:async()=>({runId:'a'})},
  {ok:true,json:async()=>({status:'queued',newLogs:[],totalLogs:0})},
  {ok:false},
  {ok:true,json:async()=>({status:'running',newLogs:[{message:'started'}],totalLogs:1})},
  {ok:true,json:async()=>({status:'complete',newLogs:[],totalLogs:1})}];
 const env={agentTicker:'TEST',agentDate:'2026-10-11',agentProvider:'fixture',agentModel:'fixture',API_URL:'fixture',agentPollRef:{current:null},
  fetch:async()=>responses.shift(),clearInterval:id=>clears.push(id),setInterval:fn=>{tick=fn;return 7;},
  setAgentActiveRun:value=>{row=typeof value==='function'?value(row):value;},setAgentView:()=>{},fetchAgentHistory:()=>history++,
  mergeAgentSnapshot,agentRunTerminal,console:{warn:()=>{},error:()=>{}},alert:()=>{throw Error('unexpected alert');}};
 await new Function(...Object.keys(env),'return ('+body+');')(...Object.values(env))();
 assert.equal(row.status,'queued');await tick();assert.equal(row.status,'queued');assert.equal(clears.includes(7),false);
 await tick();assert.equal(row.status,'queued');assert.equal(clears.includes(7),false);assert.match(row.readError,/Status refresh failed/);
 await tick();assert.equal(row.status,'running');assert.equal(row.readError,null);await tick();assert.equal(row.status,'complete');assert.equal(history,1);assert.equal(clears.includes(7),true);
});
