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

    // Serve static assets via the assets binding
    return env.ASSETS.fetch(request);
  },
};
