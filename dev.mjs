// Charlie local dev server — fast frontend loop (no deploy needed).
//
//   node dev.mjs        →  http://127.0.0.1:3000
//
// Edits to src/app.jsx rebuild + live-reload instantly. API calls go to the
// (prod) backend, which the frontend already targets on localhost and which
// already allows CORS from :3000. Run backend locally too (Stage 2) and this
// keeps working. Nothing here is used by the Render/Cloudflare deploy.
import * as esbuild from 'esbuild';
import babel from '@babel/core';
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';

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

// 3) esbuild: bundle src/app.jsx -> dev/app.js, watch, and serve with live reload
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

const ctx = await esbuild.context({
  entryPoints: ['src/app.jsx'],
  bundle: true,
  outdir: OUT,
  format: 'iife',
  target: 'es2020',
  plugins: [babelPlugin],
  sourcemap: true,
  define: { 'process.env.NODE_ENV': '"development"' },
  logLevel: 'info',
});
await ctx.watch();
const { host, port } = await ctx.serve({ servedir: OUT, host: '127.0.0.1', port: PORT });
console.log(`\n▸ Charlie dev server:  http://${host}:${port}`);
console.log('  edit src/app.jsx → rebuild + live-reload. API → prod backend.\n');
