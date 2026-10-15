import * as React from 'react';
import {originalBlob,originalFilename} from './original-download.mjs';
export function OriginalDownload({url,filename,sha256}){
 const [busy,setBusy]=React.useState(false),[error,setError]=React.useState('');const lock=React.useRef(false);
 const download=async()=>{if(lock.current)return;lock.current=true;setBusy(true);setError('');try{
  // Use the application's authenticated fetch; credentials never appear in a URL.
  const blob=await originalBlob(await fetch(url,{signal:AbortSignal.timeout(30000)}),sha256);
  const link=document.createElement('a'),objectUrl=URL.createObjectURL(blob);
  try{link.href=objectUrl;link.download=originalFilename(filename);document.body.appendChild(link);link.click();}
  finally{link.remove();setTimeout(()=>URL.revokeObjectURL(objectUrl),1000);}
 }catch(e){setError(e.message||'Original download could not be verified.');}finally{lock.current=false;setBusy(false);}};
 return <span><button type="button" disabled={busy} onClick={download}>{busy?'Checking original…':'Download original'}</button>{error&&<small role="alert" style={{display:'block'}}>{error}</small>}</span>;
}
