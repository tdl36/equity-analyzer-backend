// Deterministic UI regression checks; all API responses are synthetic.
const fs = require('fs'),
  path = require('path'),
  {
    chromium
  } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = process.cwd();
const jsx = `import React from 'react';import{createRoot}from'react-dom/client';import{SourcePreferences}from'./src/source-preferences.jsx';import{ResearchChat}from'./src/research-chat.jsx';function Test(){const[open,setOpen]=React.useState(false);return <div className="charlie-workspace"><div className="workspace-content research-desk"><SourcePreferences api="" onChange={()=>{}}/><button onClick={()=>setOpen(true)}>Open chat</button>{open&&<ResearchChat api="" context={{ticker:'ABT',type:'review',content:'Synthetic evidence'}} onClose={()=>setOpen(false)} allowEdits={false}/>}</div></div>}createRoot(document.getElementById('root')).render(<Test/>);`;
const bundle = require('esbuild').buildSync({
  stdin: {
    contents: jsx,
    loader: 'jsx',
    resolveDir: root
  },
  bundle: true,
  write: false
}).outputFiles[0].text;
(async () => {
  const browser = await chromium.launch({
    channel: 'chrome',
    headless: true
  });
  const page = await browser.newPage();
  let errors = [];
  page.on('pageerror', e => {
    errors.push(e.message);
    console.error(e.message);
  });
  await page.route('**/*', route => {
    const url = route.request().url();
    if (url.endsWith('/')) return route.fulfill({
      contentType: 'text/html',
      body: '<div id="root"></div>'
    });
    if (route.request().method() !== 'GET') throw Error('Unexpected write');
    let data = url.includes('source-preferences') ? {
      revision: 1,
      policy: {
        mode: 'auto',
        rules: [],
        subsectors: {}
      }
    } : url.includes('source-shortlists') ? {
      shortlists: []
    } : url.includes('conversations') ? {
      conversations: [],
      messages: []
    } : {
      analysts: []
    };
    return route.fulfill({
      json: data
    });
  });
  for (const width of [1440, 768, 390, 320]) for (const theme of ['ink', 'dusk', 'oak', 'bloc', 'harbor', 'graphite', 'parchment']) {
    await page.setViewportSize({
      width,
      height: 900
    });
    await page.goto('http://localhost:9945/');
    await page.addStyleTag({
      content: fs.readFileSync('dist/tailwind.css', 'utf8')
    });
    await page.evaluate(t => document.documentElement.dataset.theme = t, theme);
    await page.addScriptTag({
      content: bundle
    });
    const field = page.getByLabel('Who chooses broker research?');
    await field.waitFor();
    const result = await field.evaluate(e => {
      const s = getComputedStyle(e);
      const rgb = v => v.match(/[\d.]+/g).slice(0, 3).map(Number);
      const lum = a => a.map(v => v / 255).map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4).reduce((n, v, i) => n + v * [.2126, .7152, .0722][i], 0);
      const a = lum(rgb(s.color)),
        b = lum(rgb(s.backgroundColor));
      return {
        contrast: (Math.max(a, b) + .05) / (Math.min(a, b) + .05),
        font: parseFloat(s.fontSize),
        right: e.getBoundingClientRect().right,
        scroll: document.documentElement.scrollWidth
      };
    });
    if (result.contrast < 4.5 || result.right > width + 1 || result.scroll > width + 1 || width <= 640 && result.font < 16) throw Error(JSON.stringify({
      width,
      theme,
      result
    }));
    await page.getByRole('button', {
      name: 'Open chat'
    }).click();
    if (!(await page.getByRole('dialog').evaluate(e => e.contains(document.activeElement)))) throw Error('Modal did not receive focus');
    await page.keyboard.press('Escape');
    await page.getByRole('dialog').waitFor({
      state: 'hidden'
    });
    if (!(await page.getByRole('button', {
      name: 'Open chat'
    }).evaluate(e => e === document.activeElement))) throw Error('Focus not restored');
  }
  if (errors.length) throw Error(errors.join('\n'));
  await browser.close();
  console.log('PASS: 28 theme/viewport combinations, readable control contrast, mobile field sizing, no overflow, dialog focus and Escape.');
})();
