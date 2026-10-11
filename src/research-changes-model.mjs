import {compareCases} from './thesis-evolution-model.mjs';
import {parseTimestamp} from './workspace-model.mjs';
const time=x=>parseTimestamp(x)?.getTime()??null;
const evidenceKey=x=>JSON.stringify([x.jobId,x.changeId,x.assumptionId,x.field,x.before,x.after]);
export function changeBrief(history,proposals,selected='review') {
 const cases=[...(history?.cases||[])].sort((a,b)=>b.revision-a.revision),latest=cases[0];
 const review=[...(history?.proposalReviews||[])].filter(r=>time(r.review?.recordedAt)!==null)
  .sort((a,b)=>time(b.review.recordedAt)-time(a.review.recordedAt))[0];
 const cutoff=selected==='review'?time(review?.review?.recordedAt):null;
 const baseline=selected==='review'?(cutoff===null?null:cases.find(r=>time(r.created_at)!==null&&time(r.created_at)<=cutoff)):cases.find(r=>String(r.revision)===String(selected));
 const at=selected==='review'?review?.review?.recordedAt:baseline?.created_at;
 const seen=new Set((baseline?.body?.evidenceLinks||[]).map(evidenceKey));
 const accepted=baseline?(latest?.body?.evidenceLinks||[]).filter(e=>!seen.has(evidenceKey(e))):[];
 const pending=(proposals||[]).filter(p=>p.target==='investment_case'&&['awaiting_approval','queued','running','failed'].includes(p.status))
  .map(p=>({...p,changedSinceBaseline:!!at&&time(p.updated_at??p.created_at)!==null&&time(p.updated_at??p.created_at)>time(at)}));
 const changes=baseline&&latest?compareCases(baseline.body,latest.body):[];
 if(baseline&&latest&&JSON.stringify(baseline.body?.operatingModel||{})!==JSON.stringify(latest.body?.operatingModel||{}))changes.push({label:'Operating model · saved inputs and provenance',before:JSON.stringify(baseline.body?.operatingModel||{},null,2),after:JSON.stringify(latest.body?.operatingModel||{},null,2)});
 return {latest,baseline,review,at,changes,accepted,pending};
}
