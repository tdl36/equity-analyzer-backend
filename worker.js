const BACKEND_URL = 'https://equity-analyzer-backend.onrender.com';

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

    // Version endpoint for auto-update checks (bypasses SW cache since worker handles it)
    if (url.pathname === '/version') {
      return new Response(JSON.stringify({ version: '2026-03-15T01' }), {
        headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
      });
    }

    // Cache-clearing utility page (visit /clear in Safari to reset the PWA)
    if (url.pathname === '/clear') {
      return new Response(`<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width"><title>Clear Cache</title></head>
<body style="font-family:system-ui;padding:40px;text-align:center;background:#111;color:#fff">
<h2>Clearing Charlie cache...</h2><p id="status">Working...</p>
<script>
(async()=>{
  const s=document.getElementById('status');
  let steps=[];
  try{
    const regs=await navigator.serviceWorker.getRegistrations();
    for(const r of regs){await r.unregister();}
    steps.push('Service workers unregistered ('+regs.length+')');
  }catch(e){steps.push('SW: '+e.message);}
  try{
    const keys=await caches.keys();
    for(const k of keys){await caches.delete(k);}
    steps.push('Caches cleared ('+keys.length+')');
  }catch(e){steps.push('Cache: '+e.message);}
  s.innerHTML=steps.join('<br>')+'<br><br><b style="color:#4ade80">Done! Close this tab, then reopen Charlie from your home screen.</b>';
})();
</script></body></html>`, {
        headers: { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' },
      });
    }

    // Serve static assets via the assets binding
    return env.ASSETS.fetch(request);
  },
};
