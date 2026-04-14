const BACKEND_URL = 'https://equity-analyzer-backend.onrender.com';
const BUILD_VERSION = '2026-04-14T01';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
          'Access-Control-Max-Age': '86400',
        },
      });
    }

    // Version endpoint for auto-update detection (bypasses service worker cache)
    if (url.pathname === '/version') {
      return new Response(JSON.stringify({ version: BUILD_VERSION }), {
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-store, no-cache, must-revalidate',
          'Access-Control-Allow-Origin': '*',
        },
      });
    }

    // Nuclear cache clear page — always served fresh from edge
    if (url.pathname === '/clear') {
      return new Response(`<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width"><title>Clearing...</title></head>
<body style="background:#0a0a0a;color:#fff;font-family:system-ui;padding:40px;text-align:center">
<h2>Clearing cache...</h2><p id="status">Working...</p>
<script>
(async()=>{
  const s=document.getElementById('status');
  try{
    const regs=await navigator.serviceWorker.getRegistrations();
    for(const r of regs) await r.unregister();
    s.textContent='Unregistered '+regs.length+' service worker(s). Clearing caches...';
    const keys=await caches.keys();
    for(const k of keys) await caches.delete(k);
    s.textContent='Cleared '+keys.length+' cache(s). Redirecting...';
    setTimeout(()=>window.location.href='/',1000);
  }catch(e){s.textContent='Error: '+e.message;}
})();
</script></body></html>`, {
        headers: {
          'Content-Type': 'text/html',
          'Cache-Control': 'no-store, no-cache, must-revalidate',
        },
      });
    }

    // Proxy API and health requests to Render backend
    if (url.pathname.startsWith('/api/') || url.pathname === '/health') {
      try {
        const backendUrl = BACKEND_URL + url.pathname + url.search;
        const backendRequest = new Request(backendUrl, {
          method: request.method,
          headers: request.headers,
          body: request.method !== 'GET' && request.method !== 'HEAD' ? request.body : undefined,
        });
        const response = await fetch(backendRequest);
        // Add CORS headers
        const newHeaders = new Headers(response.headers);
        newHeaders.set('Access-Control-Allow-Origin', '*');
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: newHeaders,
        });
      } catch (e) {
        // Backend unreachable or timed out — return a proper JSON error
        return new Response(JSON.stringify({ error: 'Backend unavailable. It may be restarting — please try again in 30 seconds.' }), {
          status: 502,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
          },
        });
      }
    }

    // Serve static assets via the assets binding
    return env.ASSETS.fetch(request);
  },
};
