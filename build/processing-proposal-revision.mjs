import {processingProposal} from './processing-proposal-status.mjs';
import {emptyDraft,validDraft} from './processing-proposal-draft.mjs';
// Copy intent only. Never carry request identity, authorization, hashes or model plans.
export function proposalRevisionDraft(value,job){
 const p=processingProposal(value,job),plan=p.plan;
 if(Object.keys(plan).some(k=>!['ticker','since','until','outputs','horizon','instruction','researchPurpose','meeting','labFullNote','visualFundamentals'].includes(k)))
  throw Error('This proposal has choices this editor cannot preserve. Review its frozen plan before preparing a new proposal.');
 if(plan.labFullNote!==undefined&&(!plan.labFullNote||typeof plan.labFullNote.allowPdfText!=='boolean'||Object.keys(plan.labFullNote).some(k=>k!=='allowPdfText')))throw Error('Saved full-note choices could not be copied completely. No draft was replaced.');
 if((plan.horizon!==undefined&&(typeof plan.horizon!=='string'||!plan.horizon.trim()))||(plan.visualFundamentals!==undefined&&typeof plan.visualFundamentals!=='boolean'))throw Error('Saved horizon or public-data choices could not be copied. No draft was replaced.');
 const receipts=new Map((job.result?.sources||[]).map(s=>[s.filename,s.sha256]));
 const selected=[],supplements=[];
 for(const source of p.sources){
  const role=p.sourceContext?.[source.documentId]?.role;
  if(receipts.get(source.filename)===source.sha256&&(role===undefined||role==='collection'))selected.push(source.filename);
  else if(role==='supplemental'&&!receipts.has(source.filename))supplements.push(source.documentId);
  else throw Error('A saved source cannot be mapped to this collection and its supplemental library. Review the frozen sources before preparing a new proposal.');
 }
 const purpose=plan.researchPurpose;
 const draft=validDraft({...emptyDraft(),selected,supplements,purpose:purpose.kind,title:purpose.event?.title||'',eventDate:purpose.event?.date||'',
  outputs:[...plan.outputs],instruction:plan.instruction||'',horizon:plan.horizon||'12–24 months',visualFundamentals:plan.visualFundamentals??false,
  fullNote:!!plan.labFullNote,allowPdfText:plan.labFullNote?.allowPdfText??false,meetingDate:plan.meeting?.date||'',meetingType:plan.meeting?.type||'management'});
 if(!draft||!draft.horizon.trim()||!selected.length)throw Error('Saved proposal choices could not be copied completely. No draft was replaced.');
 return draft;
}
