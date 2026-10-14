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

const gapLabels={
 missing_acquisition_metadata:'Acquisition details were not recorded.',
 unsupported_acquisition_kind:'The acquisition type is unsupported.',
 provider_mismatch:'The recorded provider differs from the expected provider.',
 endpoint_or_issuer_mismatch:'The recorded endpoint or issuer differs from the saved company.',
 collector_version_mismatch:'The recorded collector version is not the expected version.',
 retrieval_date_mismatch:'The acquisition time disagrees with the retained retrieval date.'
};
export function acquisitionWarnings(checks,selection){
 const sources=[['Price',selection.priceSource.id],['Earnings',selection.earningsSource.id]];
 if(checks===undefined)return sources.map(([label])=>`${label} source: Acquisition consistency checks are unavailable; provenance still requires review.`);
 if(!checks||Array.isArray(checks)||typeof checks!=='object'||Object.keys(checks).some(id=>!sources.some(([,key])=>key===id)))throw Error('Acquisition checks do not match the selected originals. Reload saved evidence.');
 return sources.map(([label,id])=>{
  const row=checks[id];
  if(!row)return `${label} source: No retained acquisition check is available.`;
  if(row.verified!==false||!['missing','mismatch','consistent_unverified'].includes(row.status)||!Array.isArray(row.gaps)||row.gaps.length>5||new Set(row.gaps).size!==row.gaps.length||row.gaps.some(code=>!Object.hasOwn(gapLabels,code))||
   (row.status==='missing'?(row.metadataDigest!==null||row.gaps.length!==1||row.gaps[0]!=='missing_acquisition_metadata'):(!/^[a-f0-9]{64}$/.test(row.metadataDigest||'')||(row.status==='consistent_unverified'?row.gaps.length!==0:row.gaps.length===0))))throw Error('Acquisition checks have an unexpected status. Reload saved evidence.');
  return `${label} source: ${row.status==='consistent_unverified'?'Recorded acquisition details are consistent, but source provenance is not verified.':row.gaps.map(code=>gapLabels[code]).join(' ')}`;
 });
}
