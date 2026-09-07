export function eventSources(manifest, topic, recorded=[]) {
  if(!manifest?.lastUpdated)return {available:false,files:[],added:[],missing:[]};
  const prefix=`Catalysts/${topic}`;
  const files=(manifest.files||[]).filter(f=>f.folder===prefix||f.folder?.startsWith(prefix+'/')).filter(f=>!/(^|\/)(processed\/|recap_|synthesis_)/i.test(f.path||f.filename||''));
  const names=new Set(files.map(f=>f.path||f.filename));
  const previous=new Set(recorded);
  return {available:true,updated:manifest.lastUpdated,files,added:[...names].filter(n=>!previous.has(n)),missing:[...previous].filter(n=>!names.has(n))};
}
