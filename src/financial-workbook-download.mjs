import {saveOriginal} from './research-originals.mjs';
export async function financialWorkbook(api,assignmentId,{fetcher=fetch}={}){
 if(!/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(assignmentId||''))throw Error('Choose a saved research assignment.');
 const r=await fetcher(`${api}/api/research/assignments/${assignmentId}/financial-history`,{signal:AbortSignal.timeout(30000)});
 if(!r.ok){let d;try{d=await r.json();}catch{}throw Error(d?.error||'Financial workbook could not be downloaded.');}
 const bytes=await r.arrayBuffer();if(bytes.byteLength<4||bytes.byteLength>2_000_000||new Uint8Array(bytes)[0]!==80||new Uint8Array(bytes)[1]!==75)throw Error('The workbook response could not be verified. No file was saved.');
 return new Blob([bytes],{type:'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'});
}
export const saveFinancialWorkbook=(blob,ticker,id)=>saveOriginal(blob,`${ticker}-financial-history-${id.slice(0,8)}.xlsx`);
