import {routeHash,validWork} from './workspace-model.mjs';

export function assignmentLabNote(job) {
 if(job.input?.collectionOnly)return null;
 const s=job.result||{},id=validWork('pipelinelab','','labnote:'+s.labNoteId)?s.labNoteId:null;
 if(!id&&!job.input?.labFullNote)return null;
 const saved=!!id&&s.labNoteStatus==='draft';
 const status=saved?'Draft saved':s.labNoteStatus==='attention'?'Needs attention':job.status==='complete'?'Receipt missing':['cancelled','stopped'].includes(job.status)?'Stopped before delivery':id?'Not yet saved':'Not yet started';
 return {saved,status,href:id?routeHash('pipelinelab',job.ticker,'','labnote:'+id):null};
}

export function reviewDocuments(job) {
 const s=job.result||{},p=s.reviewPackage,base='/api/research/assignments/'+encodeURIComponent(job.id);
 const docs=[];
 if(p?.editionId)docs.push({id:'visual',label:'One-pager',url:base+'/onepager-editions/'+encodeURIComponent(p.editionId)+'/view?layout=recovery',json:true,editionId:p.editionId});
 else {const a=s.artifacts?.find(a=>a.kind==='visual');if(a)docs.push({id:'visual',label:'One-pager',url:a.url});}
 if(p)docs.push({id:'stock_summary',label:'Stock summary',url:base+'/review-package/summary'});
 else {const a=s.artifacts?.find(a=>a.kind==='stock_summary');if(a)docs.push({id:'stock_summary',label:'Stock summary',url:a.url});}
 const thesis=p?.thesisDraftId||s.thesisDraftId||s.separateThesisProposal?.id;
 if(thesis)docs.push({id:'thesis',label:'Thesis changes',actionLabel:'Review thesis changes',description:'Compare the proposed changes with your current thesis and inspect the source evidence. Decisions remain in the thesis review screen.',href:'?thesisDraft='+encodeURIComponent(thesis)+'#view=thesisimports&ticker='+encodeURIComponent(job.ticker)});
 for(const a of s.artifacts||[])if(!['visual','stock_summary'].includes(a.kind))docs.push({id:a.kind,label:a.kind==='summary_note'?'Summary note':a.kind.replaceAll('_',' '),url:a.url});
 if(s.meetingId)docs.push({id:'meeting',label:'Meeting questions',actionLabel:'Open meeting questions',description:'Open the saved questions from this research pack.',href:'#view=meetingprep&ticker='+encodeURIComponent(job.ticker)+'&work=meeting%3A'+encodeURIComponent(s.meetingId)});
 const note=assignmentLabNote(job);
 if(note?.saved)docs.push({id:'lab_note',label:'Full Lab note',href:note.href,actionLabel:'Review full Lab note',description:'Open the saved versioned note, its research purpose and frozen sources. Review, edits, exports and revision-specific acceptance are available in Pipeline Lab. Opening it starts no generation and changes no thesis.'});
 return docs;
}

// The stable view is scoped to the saved package, never to the latest edition.
export function verifiedReviewPage(value, doc, job) {
 if(!value || value.id!==doc.editionId || value.assignmentId!==job.id ||
    !job.result?.reportId || value.runId!==job.result.reportId || value.ticker!==job.ticker ||
    !['pending','accepted'].includes(value.status) || value.layout!=='recovery' ||
    typeof value.html!=='string' || !value.html.trim())
  throw Error('This response does not match the saved edition. Reload the assignment; no other document was substituted.');
 return {html:value.html,status:value.status};
}
