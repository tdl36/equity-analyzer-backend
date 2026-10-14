import {parseTimestamp} from './workspace-model.mjs';
// Never collapse a source into another source merely because filenames match.
export function labDocuments(documents, files) {
    if (!Array.isArray(documents) || !Array.isArray(files)) throw Error('Source inventory unavailable.');
    const rows = [
        ...documents.map(d => ({...d, key: 'charlie:'+d.id, location:'Charlie', folder:'Stored originals', date:d.publishedDate||d.createdAt, dateLabel:d.publishedDate?'Published':'Added', ready:true})),
        ...files.map(d => ({...d, key:'icloud:'+JSON.stringify([d.folder||'main',d.filename]), location:'iCloud', folder:d.folder||'main', date:d.modified, dateLabel:'File modified', ready:false}))
    ];
    const seen=new Set();
    for(const row of rows){
        if(typeof row.filename!=='string'||!row.filename||seen.has(row.key)||row.key.length>1200)throw Error('Source inventory contains missing or ambiguous identities.');
        if(row.location==='Charlie'&&!row.id)throw Error('A stored source identity is missing.');
        seen.add(row.key);
    }
    const names=new Map();for(const r of rows)names.set(r.filename,(names.get(r.filename)||0)+1);
    return rows.map(r=>({...r,repeated:names.get(r.filename)>1,type:suggestType(r.filename)}));
}
export function suggestType(name) {
    const s=name.toLowerCase();
    if(/transcript|earnings.call/.test(s))return 'Transcripts';
    if(/10-k|10-q|8-k|20-f|6-k/.test(s))return 'Filings';
    if(/press.release|earnings.release/.test(s))return 'Releases';
    if(/broker.report|research.report|equity.research|price.target/.test(s))return 'Broker research';
    if(/presentation|slides|investor.day/.test(s))return 'Presentations';
    return 'Other / unclassified';
}
export function visibleLabDocuments(rows,query,group,sort) {
    return rows.filter(r=>(group==='All'||r.type===group)&&`${r.filename} ${r.folder} ${r.publisher||''}`.toLowerCase().includes(query.toLowerCase())).sort((a,b)=>{
        if(sort==='date'){
            const stamp=r=>Number.isFinite(Date.parse(r.date))?Date.parse(r.date):-Infinity;
            if(stamp(a)!==stamp(b))return stamp(a)>stamp(b)?-1:1;
        }
        return a.filename.localeCompare(b.filename)||a.key.localeCompare(b.key);
    });
}

export function recentLabDocuments(rows, days, now=Date.now()) {
    if(days==='all')return rows;
    const cutoff=now-Number(days)*86400000;
    return rows.filter(r=>Number.isFinite(Date.parse(r.date))&&Date.parse(r.date)>=cutoff&&Date.parse(r.date)<=now);
}
export function labCollectionStatus(data,ticker,now=Date.now()) {
    const snapshot=data?.snapshot;
    if(!snapshot||!Array.isArray(snapshot.policies)||!Array.isArray(snapshot.requests))return 'Collection status unavailable';
    const reported=parseTimestamp(data.updatedAt);
    if(!reported||now-reported.getTime()>180000||now<reported.getTime())return 'Mac collection report is stale — check collection recovery';
    const policy=snapshot.policies.find(p=>p.ticker===ticker);
    const open=snapshot.requests.filter(r=>r.ticker===ticker&&!['complete','cancelled'].includes(r.status));
    const issue=open.find(r=>['attention','needs_auth'].includes(r.status));
    if(issue)return `Collection needs attention · ${issue.status==='needs_auth'?'sign in to AlphaSense':'open collection recovery'}`;
    if(open.length)return `${open.length} collection request(s) pending · last reported status`;
    return policy?.lastSuccess?`Collection verified through ${policy.lastSuccess} · not a guarantee of complete coverage`:policy?'No successful collection recorded for this policy':'No recurring collection policy reported';
}
