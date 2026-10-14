export function acquisitionDetails(value){
 if(!value||value.status==='missing')return {message:'Acquisition details were not recorded for this original.'};
 if(value.status!=='recorded_unverified'||value.verified!==false)return {message:'Acquisition details are unavailable. Source provenance is unverified.'};
 const identity=/^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/;
 try{
  if(['provider','collectorVersion','observedUrl','retrievedAt'].some(key=>typeof value[key]!=='string')||/[\x00-\x20]/.test(value.observedUrl)||!/(Z|[+-]\d{2}:\d{2})$/.test(value.retrievedAt))throw Error();
  const url=new URL(value.observedUrl);
  if(!identity.test(value.provider)||!identity.test(value.collectorVersion)||value.observedUrl.length>2048||url.protocol!=='https:'||url.username||url.password||url.search||url.hash||url.port||!Number.isFinite(Date.parse(value.retrievedAt)))throw Error();
  return {message:'Acquisition recorded; source provenance still requires verification.',provider:value.provider,url:value.observedUrl,time:value.retrievedAt,version:value.collectorVersion};
 }catch{return {message:'Acquisition details are unavailable. Source provenance is unverified.'};}
}
