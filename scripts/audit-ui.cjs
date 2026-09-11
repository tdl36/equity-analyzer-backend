// Read-only full-app browser audit. See docs/ui-release-audit.md.
const fs = require('fs');
const {
  chromium
} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const dir = process.env.AUDIT_OUTPUT || '/tmp/charlie-ui-audit';
fs.mkdirSync(dir, {
  recursive: true
});
const headers = process.env.AUDIT_HEADERS ? JSON.parse(fs.readFileSync(process.env.AUDIT_HEADERS)) : {};
const base = process.env.AUDIT_URL || 'http://127.0.0.1:8788';
const views = ['today', 'companies', 'library', 'create', 'automations', 'portfolio', 'chat', 'overview', 'summary', 'research', 'meetingprep', 'slides', 'studio', 'onepager', 'formats', 'dashboard', 'review', 'pipeline', 'deepdive', 'explain', 'agents', 'analysts', 'alerts', 'feed', 'settings', 'desk'];
(async () => {
  const browser = await chromium.launch({
    channel: 'chrome',
    headless: true
  });
  const context = await browser.newContext({
    serviceWorkers: 'block'
  });
  await context.addInitScript(() => {
    if (!localStorage.getItem('charlie_auth_token')) localStorage.setItem('charlie_auth_token', 'ui-audit');
    localStorage.setItem('charlie_theme', 'dusk');
  });
  let failures = [],
    errors = [],
    blocked = [];
  await context.route('**/api/**', async route => {
    const req = route.request();
    if (req.method() !== 'GET') {
      blocked.push({
        url: req.url(),
        method: req.method()
      });
      return route.fulfill({
        status: 409,
        json: {
          error: 'Read-only audit: action not submitted'
        }
      });
    }
    try {
      const response = await route.fetch({
        url: req.url().replace(base, process.env.AUDIT_API || 'https://equity-analyzer-backend.onrender.com'),
        headers: {
          ...req.headers(),
          ...headers
        },
        timeout: 60000
      });
      if (response.status() >= 400) failures.push({
        url: req.url(),
        status: response.status(),
        error: (await response.json().catch(() => ({}))).error || 'HTTP error'
      });
      await route.fulfill({
        response,
        headers: {
          ...response.headers(),
          'access-control-allow-origin': '*'
        }
      });
    } catch (e) {
      failures.push({
        url: req.url(),
        status: 0,
        error: e.message
      });
      await route.abort();
    }
  });
  const p = await context.newPage();
  p.on('pageerror', e => errors.push(e.message));
  let results = [];
  for (const width of (process.env.AUDIT_WIDTHS || '1440,768,390,320').split(',').map(Number)) {
    await p.setViewportSize({
      width,
      height: 1000
    });
    for (const view of views) {
      errors = [];
      failures = [];
      await p.goto(base + '/?local=0#view=' + view + '&ticker=ABT');
      await p.waitForLoadState('networkidle', {
        timeout: 20000
      }).catch(() => {});
      await p.waitForTimeout(500);
      const scan = await p.evaluate(() => {
        const vis = e => e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden';
        return {
          title: document.title,
          overflow: document.documentElement.scrollWidth - innerWidth,
          wide: [...document.querySelectorAll('main *, .workspace-content *')].filter(e => vis(e) && e.getBoundingClientRect().right > innerWidth + 5 && getComputedStyle(e).position !== 'fixed').slice(0, 10).map(e => ({
            tag: e.tagName,
            cls: e.className,
            text: e.innerText?.slice(0, 70),
            width: Math.round(e.getBoundingClientRect().width)
          })),
          rawControls: [...document.querySelectorAll('input:not([type=checkbox]):not([type=radio]),textarea,select')].filter(vis).map(e => ({
            tag: e.tagName,
            type: e.type,
            bg: getComputedStyle(e).backgroundColor,
            color: getComputedStyle(e).color,
            font: getComputedStyle(e).fontSize
          })).slice(0, 8)
        };
      });
      await p.screenshot({
        path: dir + '/' + view + '-' + width + '.png',
        fullPage: true
      });
      results.push({
        view,
        width,
        ...scan,
        errors: [...errors],
        failures: [...failures]
      });
      fs.writeFileSync(dir + '/results.json', JSON.stringify(results, null, 2));
      console.log(view, width, 'overflow', scan.overflow, 'errors', errors.length, 'http', failures.length);
    }
  }
  fs.writeFileSync(dir + '/blocked.json', JSON.stringify(blocked, null, 2));
  await browser.close();
  const failed = results.filter(r => r.errors.length || r.failures.length || r.overflow > 1);
  console.log(`${results.length} screen checks; ${failed.length} require investigation. No write requests submitted.`);
  if (failed.length) process.exitCode = 1;
})();
