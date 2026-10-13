export function reviewDocuments(job) {
 const s=job.result||{},p=s.reviewPackage,base='/api/research/assignments/'+encodeURIComponent(job.id);
 const docs=[];
 if(p?.editionId)docs.push({id:'visual',label:'One-pager',url:base+'/onepager-editions/'+encodeURIComponent(p.editionId)+'/preview?layout=recovery',json:true});
 else {const a=s.artifacts?.find(a=>a.kind==='visual');if(a)docs.push({id:'visual',label:'One-pager',url:a.url});}
 if(p)docs.push({id:'stock_summary',label:'Stock summary',url:base+'/review-package/summary'});
 else {const a=s.artifacts?.find(a=>a.kind==='stock_summary');if(a)docs.push({id:'stock_summary',label:'Stock summary',url:a.url});}
 const thesis=p?.thesisDraftId||s.thesisDraftId||s.separateThesisProposal?.id;
 if(thesis)docs.push({id:'thesis',label:'Thesis changes',href:'?thesisDraft='+encodeURIComponent(thesis)+'#view=thesisimports&ticker='+encodeURIComponent(job.ticker)});
 for(const a of s.artifacts||[])if(!['visual','stock_summary'].includes(a.kind))docs.push({id:a.kind,label:a.kind==='summary_note'?'Summary note':a.kind.replaceAll('_',' '),url:a.url});
 if(s.meetingId)docs.push({id:'meeting',label:'Meeting questions',href:'#view=meetingprep&ticker='+encodeURIComponent(job.ticker)+'&work=meeting%3A'+encodeURIComponent(s.meetingId)});
 return docs;
}
