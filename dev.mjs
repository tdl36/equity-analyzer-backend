// Charlie local dev server — fast frontend loop (no deploy needed).
//
//   node dev.mjs        →  http://127.0.0.1:3000
//
// Edits to src/app.jsx rebuild + live-reload. API calls go to the local backend
// (run-local.sh) when you visit ?local=1, else the prod backend. CORS allows :3000.
// Nothing here is used by the Render/Cloudflare deploy.
//
// Uses a plain Node http server (not esbuild's built-in serve, which was hanging
// under repeated rebuilds); esbuild runs in watch mode and pushes live-reload via SSE.
import * as esbuild from 'esbuild';
import babel from '@babel/core';
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync, existsSync, statSync, createReadStream } from 'node:fs';
import { join, extname, normalize } from 'node:path';
import http from 'node:http';

const OUT = 'dev';           // dev servedir (gitignored)
const PORT = 3000;           // must be a CORS-allowed origin (see app_v3.py)
mkdirSync(OUT, { recursive: true });

// 1) Dev shell = the real index.html, minus the service worker (it caches hard
//    and fights hot reload), with asset paths localized and live-reload injected.
let html = readFileSync('index.html', 'utf8');
html = html.replace(/<script>\s*\/\/ Register service worker[\s\S]*?<\/script>/, '');   // SW registration
html = html.replace(/<script>\s*\(function\(\)\{[\s\S]*?charlie_sw_nuked[\s\S]*?<\/script>/, ''); // SW-nuke
html = html.replace(/\/dist\//g, '/');                                                   // /dist/app.js -> /app.js
html = html.replace('</body>',
  `  <script>new EventSource('/esbuild').addEventListener('change', () => location.reload());</script>\n</body>`);
writeFileSync(`${OUT}/index.html`, html);

// 2) Tailwind --watch -> dev/tailwind.css (regenerates when you use new classes)
const tw = spawn('npx',
  ['tailwindcss', '-i', 'src/tailwind-input.css', '-o', `${OUT}/tailwind.css`, '--watch'],
  { stdio: 'inherit' });
process.on('exit', () => tw.kill());

// 3) esbuild: bundle src/app.jsx -> dev/app.js in watch mode.
// Match prod's transform exactly (JSX classic runtime + block-scoping const/let->var,
// which dodges temporal-dead-zone bugs in the big app.jsx). Run Babel in-process as an
// esbuild plugin so it stays one watcher; skip node_modules (esbuild handles deps).
const babelPlugin = {
  name: 'babel',
  setup(build) {
    build.onLoad({ filter: /\.jsx?$/ }, async (args) => {
      if (args.path.includes('/node_modules/')) return;
      const res = await babel.transformFileAsync(args.path, {
        babelrc: false, configFile: false,
        presets: [['@babel/preset-react', { runtime: 'classic' }]],
        plugins: ['@babel/plugin-transform-block-scoping'],
        sourceMaps: 'inline',
      });
      return { contents: res.code, loader: 'js' };
    });
  },
};

// Live-reload: notify connected browsers each time a rebuild finishes.
const sseClients = new Set();
const reloadPlugin = {
  name: 'live-reload',
  setup(build) {
    build.onEnd((result) => {
      const errs = (result.errors || []).length;
      console.log(`[watch] rebuilt${errs ? ` with ${errs} error(s)` : ''} — ${new Date().toLocaleTimeString()}`);
      if (!errs) for (const res of sseClients) { try { res.write('event: change\ndata: 1\n\n'); } catch {} }
    });
  },
};

const ctx = await esbuild.context({
  entryPoints: ['src/app.jsx'],
  bundle: true,
  outdir: OUT,
  format: 'iife',
  target: 'es2020',
  plugins: [babelPlugin, reloadPlugin],
  sourcemap: true,
  define: { 'process.env.NODE_ENV': '"development"' },
  logLevel: 'info',
});
await ctx.watch();

// 4) Plain, reliable static server for the dev/ folder + an SSE endpoint.
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
  '.map': 'application/json', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png',
  '.ico': 'image/x-icon', '.woff2': 'font/woff2' };

http.createServer((req, res) => {
  const urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
  if (urlPath === '/esbuild') {                        // SSE live-reload channel
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
    res.write('retry: 1000\n\n');
    sseClients.add(res);
    req.on('close', () => sseClients.delete(res));
    return;
  }
  let rel = urlPath === '/' ? '/index.html' : urlPath;
  let file = join(OUT, normalize(rel).replace(/^(\.\.[/\\])+/, ''));  // prevent path traversal
  if (!existsSync(file) || !statSync(file).isFile()) file = join(OUT, 'index.html'); // SPA fallback
  res.writeHead(200, { 'Content-Type': MIME[extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
  createReadStream(file).pipe(res);
}).listen(PORT, '127.0.0.1', () => {
  console.log(`\n▸ Charlie dev server:  http://127.0.0.1:${PORT}`);
  console.log('  edit src/app.jsx → rebuild + live-reload.  Use ?local=1 for the local backend.\n');
});
