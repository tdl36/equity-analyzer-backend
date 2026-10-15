import {labSourceTime} from './pipeline-lab-model.mjs';
const calendar=value=>typeof value==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(value)&&Number.isFinite(labSourceTime(value))?value:null;
// Current metadata preview only. The server freezes and verifies actual bytes on save.
export function processingSelection({selected,supplements,collection,library,since,until}) {
 const rows=[...selected.map(name=>({role:'collection',selection:name,metadata:collection.rows.find(r=>r.filename===name)})),
  ...supplements.map(id=>({role:'supplemental',selection:id,metadata:library.rows.find(r=>r.id==='thesis_'+id)}))].map(r=>({
   ...r,key:r.role+':'+r.selection,filename:r.metadata?.filename||(r.role==='collection'?r.selection:'Stored original #'+r.selection),
   date:calendar(r.metadata?.publishedDate),
   unavailable:!!r.metadata&&(r.role==='collection'?r.metadata.availability!=='available':!r.metadata.eligible),
   hash:(r.role==='collection'?r.metadata?.receiptSha256:r.metadata?.recordedSha256)||null
  }));
 const hashes=new Map();for(const r of rows)if(typeof r.hash==='string'&&/^[a-f0-9]{64}$/i.test(r.hash)){const h=r.hash.toLowerCase();hashes.set(h,[...(hashes.get(h)||[]),r]);}
 const duplicates=[...hashes.values()].filter(group=>group.length>1);
 const dates=rows.map(r=>r.date).filter(Boolean).sort(),start=calendar(since),end=calendar(until),windowKnown=!!start&&!!end&&start<=end;
 return {rows,total:rows.length,collection:selected.length,supplemental:supplements.length,
  missing:rows.filter(r=>!r.metadata).length,unavailable:rows.filter(r=>r.unavailable).length,
  metadataPending:!collection.loaded||collection.busy||!!collection.error||(supplements.length>0&&(!library.loaded||library.busy||!!library.error)),
  dates:{known:dates.length,unknown:rows.length-dates.length,earliest:dates[0]||null,latest:dates.at(-1)||null,windowKnown,
   before:windowKnown?dates.filter(d=>d<start).length:null,within:windowKnown?dates.filter(d=>d>=start&&d<=end).length:null,after:windowKnown?dates.filter(d=>d>end).length:null},
  duplicates,unselectedCollection:collection.loaded?collection.rows.filter(r=>!selected.includes(r.filename)).length:null,
  unselectedLibrary:library.rows.filter(r=>!supplements.includes(r.id.slice(7))).length,
  libraryComplete:library.loaded&&!library.cursor&&!library.error&&!library.busy,
  hashUnknown:rows.filter(r=>typeof r.hash!=='string'||!/^[a-f0-9]{64}$/i.test(r.hash)).length};
}
