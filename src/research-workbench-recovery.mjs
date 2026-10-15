const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const uuid=v=>typeof v==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(v);
const sameKeys=(a,b)=>Object.keys(a).sort().join('|')===Object.keys(b).sort().join('|');
const today=()=>new Date().toLocaleDateString('en-CA');
export function emptyWorkDraft(kind='model'){
 return {kind,title:'',owner:'Me',rationale:'',nextAction:'',dueDate:today(),status:'open',assumptionId:'',inputs:{revenueMillions:'',sharesMillions:'',beforeMarginPct:'',afterMarginPct:'',taxPct:'',baselineEPS:'',multiple:'',referencePrice:'',currency:'USD',period:'',basis:'Adjusted diluted EPS',asOf:today()},sourceId:'',sourceHash:'',passage:'',answerId:'',answerHash:'',resolution:'unresolved',mandate:'',benchmark:'',asOf:today(),holdingPct:'0',benchmarkPct:'',reason:'valuation',valuationAssessment:'',outcome:'',reviewConditions:[],reviewDecision:'pending'};
}
// Exclude server-owned calculations, source/answer snapshots and acceptance state.
export function workDraft(body){const base=emptyWorkDraft();return Object.fromEntries(Object.keys(base).map(k=>[k,body[k]??base[k]]));}
export const workRecoveryKey=(api,ticker)=>'charlie-workbench-draft-v1:'+JSON.stringify([api,ticker]);
function validDraft(d){
 const base=emptyWorkDraft();if(!object(d)||!sameKeys(d,base)||!['model','followup','underweight'].includes(d.kind)||!['open','reviewed','closed'].includes(d.status))return false;
 for(const k of Object.keys(base)){
  if(['inputs','reviewConditions','sourceId','answerId'].includes(k))continue;
  if(typeof d[k]!=='string'||d[k].length>20000)return false;
 }
 for(const k of ['sourceId','answerId'])if(d[k]!==''&&(!Number.isSafeInteger(d[k])||d[k]<1))return false;
 if(!object(d.inputs)||!sameKeys(d.inputs,base.inputs)||Object.values(d.inputs).some(v=>!(typeof v==='string'&&v.length<=1000)&&!(typeof v==='number'&&Number.isFinite(v))))return false;
 if(!Array.isArray(d.reviewConditions)||d.reviewConditions.length>12||d.reviewConditions.some(c=>!object(c)||!sameKeys(c,{id:0,category:0,trigger:0,state:0,evidence:0,sourceReference:0})||!uuid(c.id)||Object.values(c).some(v=>typeof v!=='string'||v.length>20000)))return false;
 return true;
}
export function validWorkRecovery(v,ticker){
 if(!object(v)||!sameKeys(v,{version:0,ticker:0,caseRevision:0,record:0,draft:0,pending:0})||v.version!==1||v.ticker!==ticker||!Number.isSafeInteger(v.caseRevision)||v.caseRevision<0||!validDraft(v.draft))return false;
 if(v.record!==null&&(!object(v.record)||!sameKeys(v.record,{id:0,revision:0})||!uuid(v.record.id)||!Number.isSafeInteger(v.record.revision)||v.record.revision<1))return false;
 const p=v.pending;
 if(p!==null){
  if(!object(p)||!sameKeys(p,{requestId:0,id:0,revision:0,body:0})||!uuid(p.requestId)||!uuid(p.id)||p.revision!==(v.record?.revision||0)||(v.record&&p.id!==v.record.id)||!object(p.body)||!sameKeys(p.body,{...v.draft,caseRevision:0})||p.body.caseRevision!==v.caseRevision)return false;
  for(const k of Object.keys(v.draft))if(JSON.stringify(p.body[k])!==JSON.stringify(v.draft[k]))return false;
 }
 return true;
}
export function readWorkRecovery(storage,key,ticker){
 const raw=storage.getItem(key);if(raw===null)return null;
 if(raw.length>1000000)throw Error('Workbench recovery is too large. Keep this tab open.');
 let v;try{v=JSON.parse(raw);}catch{throw Error('Workbench recovery is unreadable. The local copy has been retained.');}
 if(!validWorkRecovery(v,ticker))throw Error('Workbench recovery does not match this company or work revision. The local copy has been retained.');
 return v;
}
export function writeWorkRecovery(storage,key,v){
 if(!validWorkRecovery(v,v.ticker))throw Error('Workbench recovery could not be validated.');
 const raw=JSON.stringify(v);if(raw.length>1000000)throw Error('Workbench recovery is too large.');
 storage.setItem(key,raw);if(storage.getItem(key)!==raw)throw Error('Workbench recovery could not be verified.');
}
export function clearWorkRecovery(storage,key){storage.removeItem(key);if(storage.getItem(key)!==null)throw Error('Workbench recovery could not be cleared.');}
