import {parseTimestamp,validWork} from './workspace-model.mjs';
export function activityRows(data,filter='active',query='') {
 const q=query.trim().toLowerCase();
 return (data?.items||[]).filter(r=>(filter==='all'||(filter==='focus'?['blocked','review'].includes(workLane(r)):['processing','waiting','review','blocked'].includes(filter)?workLane(r)===filter:r.bucket===filter))&&(r.ticker+' '+r.title+' '+r.step).toLowerCase().includes(q));
}
export function activityAge(value,now=Date.now()) {
 const date=parseTimestamp(value);if(!date)return 'Time unavailable';
 const minutes=Math.max(0,Math.floor((now-date.getTime())/60000));
 return minutes<1?'Just now':minutes<60?`${minutes}m ago`:minutes<1440?`${Math.floor(minutes/60)}h ${minutes%60}m ago`:`${Math.floor(minutes/1440)}d ago`;
}
export function macConnected(value,now=Date.now()){const t=parseTimestamp(value);return !!t&&now-t.getTime()>=0&&now-t.getTime()<180000;}

export function collectionHealth(data,now=Date.now()) {
 const age=value=>{const t=parseTimestamp(value);return t?now-t.getTime():null;};
 const fresh=(value,limit)=>{const a=age(value);return a!==null&&a>=0&&a<limit;};
 const current=fresh(data?.asOf,180000),missing=data?.unavailable||[];
 const mac=current&&!missing.includes('Mac status')&&macConnected(data?.agentLastSeen,now);
 const report=current&&!missing.includes('Mac status')&&fresh(data?.collectionLastSeen,300000);
 const delivery=data?.collectionDelivery;
 const known=current&&!missing.includes('Collection delivery')&&Number.isInteger(delivery?.pending)&&delivery.pending>=0;
 const pending=known?delivery.pending:null;
 const timely=known&&(pending===0||fresh(delivery.oldestPendingAt,300000));
 return [
  {key:'mac',label:'Mac agent',ok:mac,value:mac?'Connected':'Not confirmed',detail:mac?'The agent is reporting. Browser availability and AlphaSense sign-in are checked during collection.':'Check that the Mac is awake, online and its Charlie agent is running.'},
  {key:'report',label:'Collection updates',ok:report,value:report?'Current':'Stale or unavailable',detail:report?'The Mac collection report is current. This does not mean every collection is complete.':'Check collection sync on the Mac, even if the agent is connected.'},
  {key:'delivery',label:'Requests reaching Mac',ok:timely,value:!known?'Unknown':pending===0?'No pending delivery':`${pending} awaiting receipt`,detail:!known?'Delivery could not be verified. Refresh Activity before submitting again.':pending===0?'Cloud collection commands have been acknowledged. Browser collection may still be waiting.':timely?'Waiting for the Mac to acknowledge queued collection commands.':'A collection command has not been acknowledged within five minutes, or its time is unknown. Check collection controls and Mac sync before resubmitting.'}
 ];
}

// These lanes describe saved state, never infer that a model or browser is running.
export function workLane(row) {
 if(row.bucket==='older')return 'older';
 if(row.bucket==='recent')return 'recent';
 if(['needs_review','awaiting_approval','proposed','detected'].includes(row.status))return 'review';
 if(row.bucket==='attention')return 'blocked';
 if(['queued','pending','waiting','collection_queued'].includes(row.status))return 'waiting';
 if(row.stale)return 'blocked';
 return 'processing';
}
function meetingWork(row) {return row.id?.startsWith('mp_jobs:')&&['pipeline','command_meeting'].includes(row.workflow);}
function meetingLink(row) {return row.meetingLinkState==='linked'&&Number.isInteger(row.meetingId)?validWork('meetingprep','','meeting:'+row.meetingId):'';}
export function nextAction(row) {
 if(meetingWork(row)&&!meetingLink(row))return (row.meetingLinkState==='unavailable'?'The saved meeting link could not be checked. Refresh Activity.':row.meetingLinkState==='mismatch'?'The saved job and meeting belong to different companies. The meeting link is withheld.':'This job has no verified saved meeting link.')+' Open the workflow to inspect retained work; do not submit a duplicate.';
 const lane=workLane(row);
 if(row.pendingStart)return row.startRecovery?(row.bucket==='attention'?'Automatic start recovery is enabled but this job is delayed. Check backend worker availability without submitting another run.':'Waiting for its first execution. Charlie can recover this saved job after a backend restart; do not submit a duplicate.'):'This job has not reserved an execution attempt. Automatic start recovery is unavailable; inspect the saved job before resubmitting.';
 if(row.executionUnconfirmed)return 'Reconcile the saved run with backend execution and provider outcome. Do not submit another paid run until its outcome is known.';
 if(row.workflow==='lab_note')return row.status==='needs_review'?'Review the saved full-note revision, its purpose and frozen sources. Acceptance is a separate investor action.':row.status==='complete'?'Open the accepted note revision. Acceptance does not change the current thesis.':row.status==='attention'?'Inspect the saved note and call outcome before resuming. Do not repeat an uncertain paid attempt.':row.status==='queued'?'Waiting for the existing full-note worker. Do not submit a duplicate.':'Open saved note progress; this view does not resume generation.';
 if(row.workflow==='research_decision')return 'Inspect the recorded issue, then save a follow-up review or a new review date. No thesis or portfolio action is automatic.';
 const intakeActions={paused:'Review automatic source intake settings before enabling comparisons.',credentials:'Add the required server-side key in Settings; never send credentials in chat.',budget:'Inspect the monthly budget in Settings before changing it.',daily_limit:'Pending sources are retained. Wait for the UTC reset or deliberately change the daily limit.',baseline:'Save a thesis for this company before enabling automatic comparison.',review:'Open evidence proposals and record your review of the existing proposal.',in_progress:'Inspect the existing company comparison before submitting another run.',worker:'No decision is currently identified. The worker will recheck eligibility before dispatch.'};
 if(row.waitCode&&intakeActions[row.waitCode])return intakeActions[row.waitCode];
 if(row.status==='needs_auth')return 'Complete sign-in or MFA directly in the source app, then resume collection.';
 if(lane==='review')return row.workflow==='catalyst_signal'?'Review the detected event and source choices before requesting collection.':'Review the saved draft or proposal. Charlie will not approve it for you.';
 if(lane==='blocked')return row.stale&&row.bucket==='active'?'Check the last saved step and dependency before retrying. A quiet job may still be running.':'Open the saved issue and retained progress. Confirm any uncertain paid call before retrying.';
 if(lane==='waiting'&&row.workflow==='auto_evidence_intake')return 'Check automation settings, daily limits and pending proposals for this company.';
 if(lane==='waiting'&&['collection_queued','queued'].includes(row.status))return 'Check the collection delivery panels when this work requires Mac collection.';
 if(lane==='waiting')return 'Waiting for a dependency or available worker. Open the workflow for its saved state.';
 if(lane==='processing')return 'No decision requested. Charlie is reporting work in progress.';
 if(lane==='older')return 'Inspect retained unfinished work before restarting or submitting a duplicate.';
 return 'Open the saved output; completion does not mean investor approval.';
}
export function workOverview(data) {
 const counts={blocked:0,review:0,processing:0,waiting:0,recent:0,older:0};
 for(const row of data?.items||[])counts[workLane(row)]++;
 return counts;
}
export function focusRows(data) {
 const rank={blocked:0,review:1,processing:2,waiting:3,recent:4,older:5};
 return [...(data?.items||[])].filter(r=>!['older','recent'].includes(workLane(r)))
  .sort((a,b)=>rank[workLane(a)]-rank[workLane(b)]||(a.title==='End-to-end research'?0:1)-(b.title==='End-to-end research'?0:1));
}

export function activityDestination(row) {
 if(meetingWork(row)){
  const work=meetingLink(row);
  return work?{view:'meetingprep',section:'',work}:row.workflow==='command_meeting'?{view:'desk',section:'command'}:{view:'meetingprep',section:''};
 }
 if(row.workflow==='lab_note')return {view:'pipelinelab',section:'',...workLink(row,'pipelinelab','','labnote')};
 if(row.workflow==='evidence_amendment')return {view:'desk',section:'evidence',...workLink(row,'desk','evidence','amendment')};
 if(row.workflow==='research_decision')return {view:'stockanalysis',section:'reviews',...workLink(row,'stockanalysis','reviews','decision')};
 const sections={auto_evidence_intake:'evidence',evidence_amendment:'evidence',catalyst_signal:'collection',catalyst_watch:'collection',collection_control:'collection',command_meeting:'command',agent_runs:'runs'};
 if(row.id?.startsWith('collection:'))return {view:'desk',section:'collection',...workLink(row,'desk','collection','collection')};
 if(sections[row.workflow])return {view:'desk',section:sections[row.workflow]};
 const cases={company_research_runs:'research',investment_committee_runs:'committee',stock_analysis_runs:'stock-analysis',research_assignment:'stock-analysis'};
 if(cases[row.workflow])return {view:'stockanalysis',section:cases[row.workflow],...workLink(row,'stockanalysis',cases[row.workflow],({research_assignment:'assignment',stock_analysis_runs:'analysis',company_research_runs:'research',investment_committee_runs:'committee'})[row.workflow])};
 return {view:row.view,section:''};
}

function workLink(row,view,section,kind) {
 const work=validWork(view,section,kind+':'+row.jobId);
 return work?{work}:{};
}

// Keep finished coordinating assignments discoverable on Home without displacing
// stalled work or implying investor approval. Backend owns the recent time window.
export function recentAssignments(data) {
 const seen=new Set();
 return [...(data?.items||[])].filter(row=>row.workflow==='research_assignment' &&
   row.bucket==='recent' && ['complete','completed'].includes(row.status) &&
   activityDestination(row).work)
  .sort((a,b)=>(parseTimestamp(b.updatedAt)?.getTime()||0)-(parseTimestamp(a.updatedAt)?.getTime()||0))
  .filter(row=>{if(seen.has(row.jobId))return false;seen.add(row.jobId);return true;}).slice(0,3);
}

// Home and Activity share one server-filtered agenda, including failed reads.
export function meetingAgendaState(data) {
 const agenda=data?.meetingAgenda;
 const day=value=>typeof value==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(value)&&Number.isFinite(Date.parse(value))&&new Date(value).toISOString().slice(0,10)===value;
 if(data?.unavailable?.includes('Meeting preparation agenda')||!agenda||!day(agenda.from)||!day(agenda.through)||Date.parse(agenda.through)-Date.parse(agenda.from)!==6*86400000||typeof agenda.truncated!=='boolean'||!Array.isArray(agenda.items)||agenda.items.length>50)return null;
 const seen=new Set();
 for(const m of agenda.items){
  if(!m||!Number.isSafeInteger(m.id)||!validWork('meetingprep','','meeting:'+m.id)||seen.has(m.id)||typeof m.ticker!=='string'||!m.ticker.trim()||!day(m.date)||m.date<agenda.from||m.date>agenda.through||typeof m.prepared!=='boolean'||(m.type!=null&&typeof m.type!=='string'))return null;
  seen.add(m.id);
 }
 return agenda;
}
