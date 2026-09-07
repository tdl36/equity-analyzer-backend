import test from 'node:test';
import assert from 'node:assert/strict';
import staging from '../../scripts/version-frontend-assets.cjs';
test('release HTML pins both assets without changing external links',()=>{
 const html='<link href="/dist/tailwind.css"><script src="/dist/app.js?v=old"></script><a href="https://example.org">Source</a>';
 const next=staging.versionHtml(html,'2026-09-07T02');
 assert.ok(next.includes('/dist/tailwind.css?v=2026-09-07T02'));
 assert.ok(next.includes('/dist/app.js?v=2026-09-07T02'));
 assert.ok(next.includes('https://example.org'));
 assert.equal(staging.versionHtml(next,'2026-09-07T02'),next);
});
