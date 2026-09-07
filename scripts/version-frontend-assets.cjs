// Give every release distinct asset URLs, including through intermediary caches.
const fs = require('node:fs');
const path = require('node:path');
function versionHtml(html, version) {
  if (!/^[A-Za-z0-9-]+$/.test(version)) throw new Error('Invalid frontend build version');
  return html.replace(/(\/dist\/(?:app\.js|tailwind\.css))(?:\?[^"']*)?(?=["'])/g, `$1?v=${version}`);
}
module.exports = {versionHtml};
if (require.main === module) {
  const root = path.resolve(__dirname, '..');
  const version = fs.readFileSync(path.join(root, 'worker.js'), 'utf8').match(/const BUILD_VERSION = '([^']+)'/)[1];
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  fs.writeFileSync(path.join(root, '.worker-assets/index.html'), versionHtml(html, version));
}
