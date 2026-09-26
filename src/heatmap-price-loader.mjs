// Session-only market cache. Never stores holdings, weights, or private research.
const cache=new Map();
const ttl=15*60*1000;
export async function loadHeatmapPrices({api,tickers,period,signal,request,onUpdate,onProgress}) {
 const key=t=>JSON.stringify([api,period,t]);
 const cached={},pending=[];
 for(const t of tickers){const item=cache.get(key(t));if(item&&Date.now()-item.at<ttl)cached[t]=item.quote;else pending.push(t);}
 let completed=tickers.length-pending.length,failed=0,cursor=0;
 const publish=quotes=>{const dates=Object.values(quotes).map(q=>q.fetchedAt).filter(Boolean).sort();onUpdate({quotes,provider:'Yahoo Finance via yfinance',fetchedAt:dates[0]||new Date().toISOString()});};
 if(signal.aborted)return 0;
 if(completed)publish(cached);
 onProgress(completed);
 async function worker(){
  while(cursor<pending.length&&!signal.aborted){
   const start=cursor;cursor+=20;const batch=pending.slice(start,start+20);
   try{
    const d=await request(api+'/api/portfolio/heatmap/returns',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tickers:batch,period}),signal});
    if(signal.aborted)return;
    const quotes={};
    for(const t of batch){const q={...(d.quotes?.[t]||{changePct:null,issue:'Market data unavailable'})};q.fetchedAt=q.fetchedAt||d.fetchedAt;quotes[t]=q;
     const at=Date.parse(q.fetchedAt);
     if(Number.isFinite(q.changePct)&&Number.isFinite(at)&&at<=Date.now()){
      if(cache.size>=24000)cache.delete(cache.keys().next().value);
      cache.set(key(t),{at,quote:q});
     }
    }
    publish(quotes);
   }catch(e){if(signal.aborted)return;failed+=batch.length;}
   completed+=batch.length;onProgress(completed);
  }
 }
 await Promise.all(Array.from({length:Math.min(3,Math.ceil(pending.length/20))},worker));
 return failed;
}
