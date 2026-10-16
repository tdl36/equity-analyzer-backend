export function checkedAnswer(row,meetingId,id,question){
 if(!row||row.id!==id||row.meeting_id!==meetingId||row.question!==question||!(row.response_notes===null||typeof row.response_notes==='string')||typeof row.status!=='string')throw Error('The saved answer identity could not be verified. Your draft is retained.');
 return row;
}
export const answerBaseline=a=>({responseNotes:a.response_notes,status:a.status});
export const answerPayload=(a,text)=>({meetingId:a.meeting_id,question:a.question,responseNotes:text,status:'answered',expected:answerBaseline(a)});
export function answerReceipt(data,base,payload){
 const row=checkedAnswer(data?.answer,base.meeting_id,base.id,base.question);
 if(typeof data.replayed!=='boolean'||row.response_notes!==payload.responseNotes||row.status!==payload.status)throw Error('The saved answer receipt did not match your submission. Check the saved answer before retrying.');
 return row;
}
