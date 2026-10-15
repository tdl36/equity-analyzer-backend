const outputs=new Set(['summary_note','stock_summary','thesis','visual','meeting_prep']);
export function emptyDraft(){return {version:1,selected:[],supplements:[],purpose:'thesis_review',title:'',eventDate:'',outputs:['thesis'],instruction:'',fullNote:false,allowPdfText:false,meetingDate:'',meetingType:'management'};}
export const draftKey=(api,id)=>'charlie-processing-authoring:'+api+':'+id;
export function validDraft(value){
 if(!value||typeof value!=='object'||Array.isArray(value))return null;
 const base=emptyDraft();if(Object.keys(value).sort().join('|')!==Object.keys(base).sort().join('|')||value.version!==1)return null;
 const list=(v,test)=>Array.isArray(v)&&v.length<=1000&&new Set(v).size===v.length&&v.every(test);
 const text=(v,max)=>typeof v==='string'&&v.length<=max;
 const date=v=>v===''||(typeof v==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(v));
 if(!list(value.selected,v=>text(v,2048)&&v.length>0)||!list(value.supplements,v=>typeof v==='string'&&/^[1-9][0-9]{0,18}$/.test(v))||value.selected.length+value.supplements.length>1000)return null;
 if(!['thesis_review','event_note'].includes(value.purpose)||!text(value.title,200)||!date(value.eventDate)||!text(value.instruction,1800)||!date(value.meetingDate)||!['management','expert','conference','other'].includes(value.meetingType))return null;
 if(!list(value.outputs,v=>outputs.has(v))||typeof value.fullNote!=='boolean'||typeof value.allowPdfText!=='boolean')return null;
 if(value.purpose==='event_note'&&value.outputs.some(v=>!['summary_note','meeting_prep'].includes(v)))return null;
 return JSON.parse(JSON.stringify(value));
}
export function readDraft(storage,key){
 try{const raw=storage.getItem(key);if(!raw)return {value:null,error:''};if(raw.length>3000000)throw Error();const value=validDraft(JSON.parse(raw));if(!value)throw Error();return {value,error:''};}
 catch{return {value:null,error:'Unsaved choices could not be restored. The saved processing proposals are unaffected.'};}
}
export function writeDraft(storage,key,value){
 const clean=validDraft(value);if(!clean)throw Error('Unsaved choices are invalid. Review them before saving.');
 if(JSON.stringify(clean)===JSON.stringify(emptyDraft()))storage.removeItem(key);
 else storage.setItem(key,JSON.stringify(clean));
}

// Preserve the exact submitted bytes/choices for replay; never repair a damaged
// recovery record into a new request or silently discard an uncertain submission.
export function readSubmission(storage,key){
 try {
  const raw=storage.getItem(key);if(!raw)return {value:null,error:''};
  if(raw.length>3000000)throw Error();
  const value=JSON.parse(raw),p=value?.plan,purpose=p?.researchPurpose;
  const object=v=>!!v&&typeof v==='object'&&!Array.isArray(v);
  if(!object(value)||!object(p)||!object(purpose)||
   !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value.requestId)||
   Object.keys(value).some(k=>!['requestId','filenames','supplementalIds','plan'].includes(k))||
   Object.keys(p).some(k=>!['researchPurpose','outputs','instruction','horizon','labFullNote','meeting'].includes(k))||
   (p.horizon!==undefined&&(typeof p.horizon!=='string'||p.horizon.length>100))||
   (p.labFullNote!==undefined&&(!object(p.labFullNote)||Object.keys(p.labFullNote).some(k=>k!=='allowPdfText')||typeof p.labFullNote.allowPdfText!=='boolean'))||
   (p.meeting!==undefined&&(!object(p.meeting)||Object.keys(p.meeting).some(k=>!['date','type'].includes(k)))))throw Error();
  const draft=validDraft({...emptyDraft(),selected:value.filenames,supplements:value.supplementalIds??[],purpose:purpose.kind,
   title:purpose.event?.title??'',eventDate:purpose.event?.date??'',outputs:p.outputs,instruction:p.instruction??'',
   fullNote:p.labFullNote!==undefined,allowPdfText:p.labFullNote?.allowPdfText??false,
   meetingDate:p.meeting?.date??'',meetingType:p.meeting?.type??'management'});
  if(!draft||!draft.selected.length||!draft.outputs.length)throw Error();
  return {value,error:''};
 }catch{return {value:null,error:'Local submission recovery is unavailable or damaged. Saved server proposals can still be reviewed. Restore browser storage and refresh before creating or retrying a submission; the local record has not been discarded.'};}
}
