#!/bin/bash
# Frontend build: Babel transpiles JSX + const/let → var, then esbuild bundles.
# Do NOT rename (build.sh is Render's Python build).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p dist build

echo "[1/3] Babel transpile..."
# -x .jsx because default doesn't pick .jsx
npx babel src --out-dir build --extensions .jsx,.js 2>&1 | tail -3

cp src/*.mjs build/

echo "[2/3] esbuild bundle..."
npx esbuild build/app.js \
  --bundle \
  --minify \
  --sourcemap \
  --outfile=dist/app.js \
  --define:process.env.NODE_ENV='"production"' \
  --target=es2017 \
  --format=iife 2>&1 | tail -3

echo "[3/3] Tailwind CSS..."
npx tailwindcss -i src/tailwind-input.css -o dist/tailwind.css --minify 2>&1 | tail -2

ls -lh dist/app.js dist/tailwind.css

# Publish an explicit set of website assets, never the repository root.
mkdir -p .worker-assets/dist .worker-assets/static/assets
cp index.html manifest.json service-worker.js icon-*.png .worker-assets/
cp dist/app.js dist/tailwind.css .worker-assets/dist/
cp static/assets/*.png .worker-assets/static/assets/
node scripts/version-frontend-assets.cjs
