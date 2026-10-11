import test from 'node:test';
import assert from 'node:assert/strict';
import {agentRunPending,agentRunTerminal,agentRunHeading,mergeAgentSnapshot,savedSubmission,selectedAgentRun,rememberAgentRun,observeAgentRun} from '../../src/agent-run-state.mjs';
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

test('actual submission blocks double clicks and remembers the confirmed run',async()=>{
 const {readFileSync}=await import('node:fs');
 const source=readFileSync(new URL('../../src/app.jsx',import.meta.url),'utf8'),prefix='const startAgentRun = ';
 const body=source.slice(source.indexOf(prefix)+prefix.length,source.indexOf('const saveAgentRun =')).trim().replace(/;$/,'');
 let calls=0,row,remembered;
 const env={agentActiveRun:null,agentSubmitRef:{current:false},setAgentSubmitting:()=>{},setAgentSubmitError:e=>{if(e)throw Error(e);},
 sessionStorage:{removeItem:()=>{}},savedSubmission:()=>({key:'fixture',id:'a'}),rememberAgentRun:(_s,id,ticker)=>{remembered={id,ticker};},
 agentRunPending,agentTicker:'TEST',agentDate:'2026-10-11',agentProvider:'fixture',agentModel:'fixture',API_URL:'fixture',
 fetch:async()=>{calls++;return {ok:true,json:async()=>({runId:'a'})};},setAgentActiveRun:value=>{row=value;},setAgentView:()=>{}};
 const start=new Function(...Object.keys(env),'return ('+body+');')(...Object.values(env));
 await Promise.all([start(),start()]);assert.equal(calls,1);assert.equal(row.status,'queued');assert.deepEqual(remembered,{id:'a',ticker:'TEST'});
});

test('submission identity survives reload and is bound to frozen settings',()=>{
 const items=new Map(),storage={getItem:k=>items.get(k),setItem:(k,v)=>items.set(k,v)};
 const id='00000000-0000-4000-8000-000000000173',payload={ticker:'TEST',date:'2026-10-11',model:'pinned'};
 const first=savedSubmission(storage,payload,()=>id);
 assert.deepEqual(savedSubmission(storage,{...payload},()=>{throw Error('must reuse');}),first);
 assert.notEqual(savedSubmission(storage,{...payload,model:'changed'},()=>id).key,first.key);
 assert.throws(()=>savedSubmission({getItem:()=>null,setItem:()=>{throw Error('storage unavailable');}},payload,()=>id),/storage unavailable/);
});


test('same-tab selection restores identity without cached report or execution claim',()=>{
 const items=new Map(),storage={getItem:k=>items.get(k),setItem:(k,v)=>items.set(k,v)},id='00000000-0000-4000-8000-000000000174';
 rememberAgentRun(storage,id,'TEST');const row=selectedAgentRun(storage);
 assert.equal(row.runId,id);assert.equal(row.ticker,'TEST');assert.equal(row.status,'restoring');assert.equal(agentRunPending(row.status),true);assert.deepEqual(row.logs,[]);
 assert.equal(selectedAgentRun({getItem:()=>'{bad'}),null);
});

test('observer reads immediately, survives queued/errors and cancels on completion',async()=>{
 let tick,clears=0,completed=0,errors=0;const snapshots=[],responses=[{status:'queued',newLogs:[],totalLogs:0},null,{status:'running',newLogs:[{}],totalLogs:1},{status:'complete',newLogs:[],totalLogs:1}],urls=[];
 const stop=observeAgentRun({api:'fixture',runId:'a',interval:fn=>{tick=fn;return 1;},cancel:()=>clears++,
 fetcher:async url=>{urls.push(url);const value=responses.shift();return {ok:!!value,json:async()=>value};},onSnapshot:(s,first)=>snapshots.push([s.status,first]),onError:()=>errors++,onComplete:()=>completed++});
 const flush=()=>new Promise(resolve=>setImmediate(resolve));await flush();await tick();await tick();await tick();
 assert.deepEqual(snapshots,[['queued',true],['running',false],['complete',false]]);assert.equal(errors,1);assert.equal(completed,1);assert.equal(clears,1);assert.match(urls.at(-1),/since=1$/);stop();
});

test('leaving cancels old responses; returning restarts from saved state without POST',async()=>{
 let resolve,updates=0,signal;
 const stop=observeAgentRun({api:'fixture',runId:'a',interval:()=>1,cancel:()=>{},fetcher:(_url,options)=>{signal=options.signal;return new Promise(r=>resolve=r);},onSnapshot:()=>updates++,onError:()=>updates++,onComplete:()=>updates++});
 stop();assert.equal(signal.aborted,true);resolve({ok:true,json:async()=>({status:'running',newLogs:[],totalLogs:0})});await new Promise(r=>setImmediate(r));assert.equal(updates,0);
 let requested;
 const returned=observeAgentRun({api:'fixture',runId:'a',interval:()=>2,cancel:()=>{},fetcher:async(url,options)=>{requested={url,options};return {ok:true,json:async()=>({status:'complete',newLogs:[],totalLogs:0})};},onSnapshot:()=>updates++,onError:()=>{},onComplete:()=>{}});
 await new Promise(r=>setImmediate(r));assert.match(requested.url,/since=0$/);assert.equal(requested.options.method,undefined);assert.equal(updates,1);returned();
});

test('batch screen reuses the saved identity and blocks simultaneous submission',async()=>{
 const {readFileSync}=await import('node:fs'),source=readFileSync(new URL('../../src/app.jsx',import.meta.url),'utf8'),prefix='const startAgentBatchRun = ';
 const body=source.slice(source.indexOf(prefix)+prefix.length,source.indexOf('const startDeepResearch =')).trim().replace(/;$/,'');
 let calls=0,removed=false,payload;
 const env={agentBatchTickers:['ONE','TWO'],agentBatchSubmitRef:{current:false},setAgentBatchSubmitting:()=>{},agentDate:'2026-10-11',agentProvider:'fixture',agentModel:'pinned',API_URL:'fixture',sessionStorage:{removeItem:()=>removed=true},savedSubmission:()=>({id:'batch-id',key:'batch-key'}),
 fetch:async(_url,options)=>{calls++;payload=JSON.parse(options.body);return {ok:true,json:async()=>({batchId:'batch-id',count:2})};},alert:()=>{},setAgentView:()=>{},fetchAgentHistory:()=>{}};
 const start=new Function(...Object.keys(env),'return ('+body+');')(...Object.values(env));await Promise.all([start(),start()]);
 assert.equal(calls,1);assert.equal(payload.requestId,'batch-id');assert.equal(removed,true);
 env.fetch=async()=>({ok:false});removed=false;await new Function(...Object.keys(env),'return ('+body+');')(...Object.values(env))();assert.equal(removed,false);
});
