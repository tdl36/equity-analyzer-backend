// A displayed proposal is actionable only after a complete, scoped status read.
const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const uuid=v=>typeof v==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(v);
const hash=v=>typeof v==='string'&&/^[0-9a-f]{64}$/.test(v);
const text=v=>typeof v==='string'&&v.length>0;
const unique=v=>new Set(v).size===v.length;
const outputs=new Set(['summary_note','stock_summary','thesis','visual','meeting_prep']);
export function processingProposal(row,job){
 const p=row?.plan,s=row?.sources,n=row?.fullNote,purpose=p?.researchPurpose;
 if(!object(row)||!uuid(row.id)||row.ticker!==job.ticker||row.collectionId!==job.id||row.revision!==1||!hash(row.digest)||!['pending','dispatched','cancelled'].includes(row.status)||
 !object(p)||p.ticker!==job.ticker||p.since!==job.input.since||p.until!==job.input.until||!object(purpose)||!['thesis_review','event_note'].includes(purpose.kind)||
 !Array.isArray(p.outputs)||!p.outputs.length||!unique(p.outputs)||p.outputs.some(o=>!outputs.has(o))||
 (purpose.kind==='event_note'&&(!object(purpose.event)||!text(purpose.event.title)||!/^\d{4}-\d{2}-\d{2}$/.test(purpose.event.date)||p.outputs.some(o=>!['summary_note','meeting_prep'].includes(o))))||
 (p.meeting!==undefined&&(!object(p.meeting)||!text(p.meeting.date)||!text(p.meeting.type)))||
 (p.instruction!==undefined&&typeof p.instruction!=='string')||
 !Array.isArray(s)||!s.length||s.length>1000||s.some(x=>!object(x)||typeof x.documentId!=='string'||!(/^[1-9][0-9]{0,18}$/).test(x.documentId)||x.reference!=='charlie:thesis_'+x.documentId||!text(x.filename)||!hash(x.sha256)||!Number.isSafeInteger(x.bytes)||x.bytes<1)||
 !unique(s.map(x=>x.documentId))||!unique(s.map(x=>x.filename))||
 (row.assignmentId!=null&&!uuid(row.assignmentId))||
 (row.status==='dispatched'&&!uuid(row.assignmentId))||
 (n!=null&&(!object(n)||!text(n.modelKey)||!Number.isSafeInteger(n.maxCalls)||n.maxCalls<1||!Number.isSafeInteger(n.maxTokens)||n.maxTokens<1||typeof n.allowPdfText!=='boolean'||!hash(n.configDigest)||!hash(n.promptSha256)||
 (n.sourcePreparation!=null&&(!object(n.sourcePreparation)||!Array.isArray(n.sourcePreparation.sourcePack)||!Number.isSafeInteger(n.sourcePreparation.batchCount)||!Array.isArray(n.sourcePreparation.warnings)||n.sourcePreparation.warnings.some(w=>!object(w)||!text(w.reference)||!text(w.issue)))))))
 throw Error('Saved processing proposal could not be verified for this company and collection. Refresh before saving or approving; previously loaded proposals remain visible.');
 return row;
}
export function processingProposalList(data,job){
 if(!object(data)||!Array.isArray(data.proposals)||data.proposals.length>50||!unique(data.proposals.map(p=>p?.id)))throw Error('Saved processing proposal list is incomplete. Refresh before saving or approving.');
 return data.proposals.map(p=>processingProposal(p,job));
}

export function processingProposalPage(data,job,{after=null,requestId=null}={}){
 const rows=processingProposalList(data,job),nextCursor=data.nextCursor??null;
 if((nextCursor!==null&&(!uuid(nextCursor)||rows.length!==50||nextCursor!==rows.at(-1)?.id||nextCursor===after))||
 (requestId&&(rows.length>1||nextCursor!==null||rows.some(p=>p.id!==requestId))))
 throw Error('Saved proposal history page could not be verified. Refresh or retry this page; loaded proposals remain visible.');
 return {rows,nextCursor};
}
export function mergeProcessingProposals(loaded,incoming){
 const rows=new Map(loaded.map(p=>[p.id,p]));
 for(const p of incoming)rows.set(p.id,p);
 return [...rows.values()];
}
