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
