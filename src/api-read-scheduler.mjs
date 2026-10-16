// Bound startup/polling bursts without retrying or delaying write requests.
export function createReadScheduler(limit = 4) {
  let active = 0;
  const waiting = [];
  const pump = () => {
    while (active < limit && waiting.length) {
      const item = waiting.shift();
      item.signal?.removeEventListener('abort', item.cancel);
      if (item.signal?.aborted) {
        item.reject(item.signal.reason || new DOMException('Aborted', 'AbortError'));
        continue;
      }
      active++;
      Promise.resolve().then(item.run).then(item.resolve, item.reject).finally(() => {
        active--;
        pump();
      });
    }
  };
  return (run, signal) => new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason || new DOMException('Aborted', 'AbortError'));
    const item = {run, signal, resolve, reject};
    item.cancel = () => {
      const index = waiting.indexOf(item);
      if (index !== -1) waiting.splice(index, 1);
      signal.removeEventListener('abort', item.cancel);
      reject(signal.reason || new DOMException('Aborted', 'AbortError'));
    };
    waiting.push(item);
    signal?.addEventListener('abort', item.cancel, {once:true});
    pump();
  });
}

// Reserve capacity for the research page the investor is opening. Startup and
// background reads must not consume its entire timeout while it is queued.
export function createApiReadScheduler() {
  const background=createReadScheduler(4), research=createReadScheduler(2);
  return (url,run,signal)=>{
    let path='';try{path=new URL(url,'https://charlie.invalid').pathname;}catch{}
    const foreground=/^\/api\/research\/(investment-case\/[^/]+|assignments|assignment-history|assignments\/[^/]+\/(review-package\/summary|onepager-editions(?:\/.*)?|artifact\/[^/]+)|stock-analysis-run\/[^/]+(?:\/rewrite\/.*)?|stock-analysis\/[^/]+)$/.test(path)||/^\/api\/thesis-imports(?:\/[^/]+)?$/.test(path);
    return (foreground?research:background)(run,signal);
  };
}

// Research reads use the already-authorized backend directly on the deployed
// frontend. Keep writes, local development and other origins unchanged.
export function researchReadUrl(url, origin, method='GET') {
  if(typeof url!=='string'||method.toUpperCase()!=='GET'||origin!=='https://charlie-deployment.tonydlee.workers.dev')return url;
  let parsed;try{parsed=new URL(url,origin);}catch{return url;}
  if(parsed.origin!==origin||!/^\/api\/(research\/|thesis-imports(?:\/|$))/.test(parsed.pathname))return url;
  return 'https://equity-analyzer-backend.onrender.com'+parsed.pathname+parsed.search;
}
