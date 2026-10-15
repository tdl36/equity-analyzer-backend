import {workDraft,validWorkRecovery} from './research-workbench-recovery.mjs';
const labels={kind:'Work type',title:'Research question / title',owner:'Owner',rationale:'Interpretation and rationale',nextAction:'Next action',dueDate:'Next action due',status:'State',assumptionId:'Assumption identity',sourceId:'Original identity',sourceHash:'Original text hash',passage:'Supporting passage',answerId:'Meeting answer identity',answerHash:'Meeting answer hash',resolution:'Answer resolution',mandate:'Portfolio / mandate',benchmark:'Benchmark',asOf:'Holdings reference date',holdingPct:'Portfolio weight %',benchmarkPct:'Benchmark weight %',reason:'Underweight reason',valuationAssessment:'Valuation assessment',outcome:'Review outcome',reviewDecision:'Underweight decision'};
const inputLabels={revenueMillions:'Revenue · millions',sharesMillions:'Diluted shares · millions',beforeMarginPct:'Baseline operating margin %',afterMarginPct:'Proposed operating margin %',taxPct:'Marginal tax rate %',baselineEPS:'Baseline EPS',multiple:'Valuation P/E',referencePrice:'Reference share price',currency:'Currency',period:'Fiscal forecast period',basis:'Accounting / EPS basis',asOf:'Model reference date'};
const display=v=>v===''||v===null||v===undefined?'Not recorded':String(v);
const conditions=v=>v.length?v.map((c,i)=>`${i+1}. ${c.category} · ${c.state}\nCondition: ${c.trigger}\nEvidence: ${c.evidence}\nSource: ${c.sourceReference}\nIdentity: ${c.id}`).join('\n\n'):'No conditions recorded';
export function latestWorkForComparison(payload,record,ticker,kind){
 if(!Array.isArray(payload?.records))throw Error('Saved work response is incomplete. Retry the comparison.');
 const rows=payload.records.filter(r=>r?.id===record.id);
 if(rows.length!==1)throw Error('This saved work could not be uniquely located. Your draft is unchanged.');
 const r=rows[0];
 if(!r.body||r.body.kind!==kind||!Number.isSafeInteger(r.revision)||r.revision<record.revision||!Number.isSafeInteger(r.body.caseRevision)||r.body.caseRevision<1)throw Error('Saved work identity or revision could not be verified. Your draft is unchanged.');
 const draft=workDraft(r.body);
 if(!validWorkRecovery({version:1,ticker,caseRevision:r.body.caseRevision,record:{id:r.id,revision:r.revision},draft,pending:null},ticker))throw Error('Saved work fields could not be verified. Your draft is unchanged.');
 return {id:r.id,revision:r.revision,caseRevision:r.body.caseRevision,draft};
}
export function workDifferences(local,saved){
 const rows=[];
 for(const [key,label] of Object.entries(labels))if(local[key]!==saved[key])rows.push({key,label,local:display(local[key]),saved:display(saved[key])});
 for(const [key,label] of Object.entries(inputLabels))if(local.inputs[key]!==saved.inputs[key])rows.push({key:'inputs.'+key,label,local:display(local.inputs[key]),saved:display(saved.inputs[key])});
 if(JSON.stringify(local.reviewConditions)!==JSON.stringify(saved.reviewConditions))rows.push({key:'reviewConditions',label:'Reconsideration conditions',local:conditions(local.reviewConditions),saved:conditions(saved.reviewConditions)});
 return rows;
}
