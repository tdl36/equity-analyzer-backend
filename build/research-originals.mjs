export async function verifiedOriginal(api,route,runId,source,{fetcher=fetch,digest=(bytes)=>crypto.subtle.digest('SHA-256',bytes)}={}) {
 if(!['stock-analysis','company','committee'].includes(route)||!source?.id||!/^[a-f0-9]{64}$/.test(source.originalHash||''))throw Error('The frozen original identity is unavailable.');
 const response=await fetcher(`${api}/api/research/${route}-run/${encodeURIComponent(runId)}/source/${encodeURIComponent(source.id)}`,{signal:AbortSignal.timeout(30000)});
 if(!response.ok) {let data;try{data=await response.json();}catch{}throw Error(data?.error||'Original download unavailable. Saved research remains intact.');}
 const bytes=await response.arrayBuffer();
 if(bytes.byteLength>60_000_000)throw Error('Original exceeds the verified download bound.');
 const hash=[...new Uint8Array(await digest(bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
 if(hash!==source.originalHash)throw Error('Downloaded bytes do not match the frozen original. The file was not saved.');
 return new Blob([bytes],{type:response.headers.get('Content-Type')||'application/octet-stream'});
}
export function saveOriginal(blob,name) {
 const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name||'original';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
}
