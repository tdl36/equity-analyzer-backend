#!/bin/bash
# Frontend build: precompile JSX (esbuild) and Tailwind CSS so the browser
# does not have to load @babel/standalone or cdn.tailwindcss.com at runtime.
#
# NOTE: this file is named build-frontend.sh (NOT build.sh) because build.sh
# is the Render backend build script for Python deps. Do not rename it.
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p dist

echo "[1/2] Bundling JSX with esbuild..."
npx esbuild src/app.jsx \
  --bundle \
  --minify \
  --sourcemap \
  --outfile=dist/app.js \
  --loader:.js=jsx \
  --loader:.jsx=jsx \
  --define:process.env.NODE_ENV=\"production\" \
  --target=es2020 \
  --format=iife

echo "[2/2] Compiling Tailwind CSS..."
npx tailwindcss -i src/tailwind-input.css -o dist/tailwind.css --minify

echo
echo "Build complete:"
ls -lh dist/app.js dist/tailwind.css | awk '{printf "  %-24s %s\n", $9, $5}'
