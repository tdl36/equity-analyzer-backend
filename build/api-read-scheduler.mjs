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
