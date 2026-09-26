// Pure functions shared by the heat map and its regression tests.
export function layout(items, x=0, y=0, width=1000, height=600) {
  const rows=items.filter(r=>Number.isFinite(r.size)&&r.size>0).slice().sort((a,b)=>b.size-a.size);
  function split(list,x,y,w,h) {
    if(!list.length)return [];
    if(list.length===1)return [{...list[0],x,y,width:w,height:h}];
    const total=list.reduce((s,r)=>s+r.size,0);
    let sum=list[0].size,i=1;
    while(i<list.length-1&&Math.abs(sum+list[i].size-total/2)<Math.abs(sum-total/2)){sum+=list[i].size;i++;}
    const ratio=sum/total;
    return w>=h?[...split(list.slice(0,i),x,y,w*ratio,h),...split(list.slice(i),x+w*ratio,y,w*(1-ratio),h)]:[...split(list.slice(0,i),x,y,w,h*ratio),...split(list.slice(i),x,y+h*ratio,w,h*(1-ratio))];
  }
  return split(rows,x,y,width,height);
}
export function groupedLayout(rows,group=true,width=1000,height=600) {
  const groups=new Map();
  rows.forEach(r=>{const key=group?(r.sector||'Unclassified'):'Portfolio'; if(!groups.has(key))groups.set(key,[]);groups.get(key).push({...r,size:Math.abs(r.weight)});});
  return layout([...groups].map(([name,children])=>({name,children,size:children.reduce((s,r)=>s+r.size,0)})),0,0,width,height)
    .map(g=>({...g,tiles:layout(g.children,g.x+2,g.y+26,Math.max(0,g.width-4),Math.max(0,g.height-28))}));
}
export const formatReturn=v=>Number.isFinite(v)?`${v>0?'+':''}${v.toFixed(2)}%`:'No data';
export function tileColor(value,accessible=false) {
  if(!Number.isFinite(value))return '#343941';
  const strength=Math.min(1,Math.abs(value)/3);
  const a=[43,48,53], b=value>=0?(accessible?[40,113,176]:[27,125,82]):(accessible?[162,87,20]:[172,48,57]);
  return `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*strength)).join(',')})`;
}
export function parseHoldings(text) {
  // RFC-style quoted CSV, including quoted commas, escaped quotes and CRLF.
  const records=[];let row=[],field='',quoted=false;
  for(let i=0;i<text.length;i++) {const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}else if((c===','||c==='\t')&&!quoted){row.push(field);field='';}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(field);if(row.some(v=>v.trim()))records.push(row);row=[];field='';}else field+=c;}
  if(quoted)throw Error('A quoted CSV field was not closed.');
  row.push(field);if(row.some(v=>v.trim()))records.push(row);
  const header=(records.shift()||[]).map(v=>v.replace(/^\uFEFF/,'').trim().toLowerCase());
  if(!header.includes('ticker')||!header.includes('weight'))throw Error('Use column headers: ticker,weight,sector,company. Weight is a percentage: 5 means 5%.');
  if(!records.length||records.length>100)throw Error('Import between 1 and 100 holdings.');
  const seen=new Set();return records.map((cells,i)=>{
    const get=k=>(cells[header.indexOf(k)]||'').trim();const ticker=get('ticker').toUpperCase(),raw=get('weight').replace(/%$/,'');const weight=raw===''?NaN:Number(raw);
    if(!/^[A-Z0-9][A-Z0-9.^=-]{0,19}$/.test(ticker)||seen.has(ticker)||!Number.isFinite(weight)||weight===0||Math.abs(weight)>1000)throw Error(`Check row ${i+2}: unique ticker and nonzero percentage weight required.`);
    seen.add(ticker);return {ticker,weight,sector:get('sector')||'Unclassified',company:get('company')||ticker};
  });
}
