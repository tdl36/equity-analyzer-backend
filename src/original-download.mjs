export async function originalBlob(response,expectedSha256){
 if(!response.ok){let message='Original unavailable. Refresh the source inventory and check permissions.';try{const data=await response.json();if(typeof data.error==='string')message=data.error;}catch{}throw Error(message);}
 const blob=await response.blob();
 if(!blob.size||blob.size>15000000)throw Error('Original is empty or exceeds the supported 15 MB download size.');
 if(expectedSha256){
  if(!/^[0-9a-f]{64}$/.test(expectedSha256))throw Error('Original receipt is invalid. Refresh before downloading.');
  const digest=await crypto.subtle.digest('SHA-256',await blob.arrayBuffer());
  const actual=Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,'0')).join('');
  if(actual!==expectedSha256)throw Error('Stored original bytes differ from this receipt. No file was downloaded. Refresh and review the changed source before processing.');
 }
 return blob;
}
export const originalFilename=name=>String(name).replace(/[<>:"/\\|?*\x00-\x1f]/g,'_').slice(0,240)||'original';
