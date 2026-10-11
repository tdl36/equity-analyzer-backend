export function questionSelection(pack,indices,meetingDate,meetingType){
 if(!pack?.fingerprint||!Array.isArray(pack.questions))throw Error('Reload saved questions before copying.');
 if(!Array.isArray(indices)||!indices.length||indices.length>10||new Set(indices).size!==indices.length||indices.some(i=>!Number.isInteger(i)||!pack.questions.some(q=>q.index===i)))throw Error('Select 1–10 saved questions.');
 if(!/^\d{4}-\d{2}-\d{2}$/.test(meetingDate)||Number.isNaN(Date.parse(meetingDate))||new Date(meetingDate).toISOString().slice(0,10)!==meetingDate)throw Error('Choose a valid meeting date.');
 if(!['management','expert','conference','other'].includes(meetingType))throw Error('Choose a meeting type.');
 return {fingerprint:pack.fingerprint,indices:[...indices].sort((a,b)=>a-b),meetingDate,meetingType};
}
export function originLink(origin){
 if(!origin||!/^\w[\w.-]{0,19}$/.test(origin.ticker||'')||!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(origin.runId||''))return '';
 return '#view=stockanalysis&ticker='+encodeURIComponent(origin.ticker)+'&section=stock-analysis&work='+encodeURIComponent('analysis:'+origin.runId);
}
