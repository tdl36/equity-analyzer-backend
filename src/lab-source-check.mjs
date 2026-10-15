// A validated server preflight is still not permission to dispatch research.
export function checkedLabSourceCheck(result,task,revision,references){
 const fail=()=>{throw Error('Source check could not be verified. Check the saved originals again before generating research.');};
 if(!result||result.task!==task||result.revision!==revision||result.checkedOnly!==true||!Array.isArray(result.verified)||result.verified.length>1000||!Array.isArray(result.blockers)||result.blockers.some(b=>typeof b!=='string'||!b))fail();
 const rows=result.verified;
 if(new Set(rows.map(r=>r?.reference)).size!==rows.length||rows.some(r=>!r||typeof r.reference!=='string'||!/^charlie:thesis_[1-9][0-9]*$/.test(r.reference)||!references.includes(r.reference)||typeof r.filename!=='string'||!r.filename||typeof r.sha256!=='string'||!/^[a-f0-9]{64}$/.test(r.sha256)||!Number.isSafeInteger(r.bytes)||r.bytes<1||r.bytes>15000000))fail();
 if(result.blockers.length){if(result.generationAvailable===true||result.digest!==null)fail();}
 else if(!rows.length||rows.length!==references.length||new Set(rows.map(r=>r.filename)).size!==rows.length||rows.reduce((n,r)=>n+r.bytes,0)>60000000||typeof result.digest!=='string'||!/^[a-f0-9]{64}$/.test(result.digest))fail();
 return result;
}
