const BACKEND_URL = 'https://equity-analyzer-backend.onrender.com';
const BUILD_VERSION = '2026-04-30T09';

// Origin allowlist. Wildcard CORS was leaking the API behind any origin and
// blocking any future cookie-auth migration. Echo back the request Origin only
// if it's on this list; otherwise omit the header (browser blocks the response).
const ALLOWED_ORIGINS = new Set([
  'https://charlie-deployment.tonydlee.workers.dev',
  'http://localhost:5173',
  'http://localhost:3000',
  'http://127.0.0.1:5173',
]);

function corsOrigin(request) {
  const origin = request.headers.get('Origin') || '';
  return ALLOWED_ORIGINS.has(origin) ? origin : null;
}

function corsHeaders(request, extra = {}) {
  const origin = corsOrigin(request);
  const h = { ...extra };
  if (origin) {
    h['Access-Control-Allow-Origin'] = origin;
    h['Vary'] = 'Origin';
  }
  return h;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: corsHeaders(request, {
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
          'Access-Control-Max-Age': '86400',
        }),
      });
    }

    // Version endpoint for auto-update detection (bypasses service worker cache)
    if (url.pathname === '/version') {
      return new Response(JSON.stringify({ version: BUILD_VERSION }), {
        headers: corsHeaders(request, {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-store, no-cache, must-revalidate',
        }),
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
        const newHeaders = new Headers(response.headers);
        const origin = corsOrigin(request);
        if (origin) {
          newHeaders.set('Access-Control-Allow-Origin', origin);
          newHeaders.append('Vary', 'Origin');
        } else {
          // Same-origin requests don't need ACAO; strip any wildcard the backend might emit.
          newHeaders.delete('Access-Control-Allow-Origin');
        }
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: newHeaders,
        });
      } catch (e) {
        return new Response(JSON.stringify({ error: 'Backend unavailable. It may be restarting — please try again in 30 seconds.' }), {
          status: 502,
          headers: corsHeaders(request, { 'Content-Type': 'application/json' }),
        });
      }
    }

    // Serve static assets via the assets binding
    return env.ASSETS.fetch(request);
  },
};


// Scheduled trigger — pings backend every 10 min to keep Render dyno warm.
// Configured via wrangler.jsonc triggers.crons.
export async function scheduled(event, env, ctx) {
  ctx.waitUntil((async () => {
    try {
      const r = await fetch(`${BACKEND_URL}/api/alerts/count`, {
        headers: { 'User-Agent': 'charlie-keepwarm/1.0' },
        signal: AbortSignal.timeout(60000),
      });
      console.log(`keepwarm: ${r.status}`);
    } catch (e) {
      console.log(`keepwarm error: ${e && e.message || e}`);
    }
  })());
}
