"""Verify app startup is not blocked by unavailable optional CDN assets."""
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='charlie-startup-') as folder:
 p=Path(folder)
 html=(ROOT/'index.html').read_text().replace('src="/dist/app.js"','src="probe.js"').replace('href="/dist/tailwind.css"',f'href="{ROOT}/dist/tailwind.css"')
 # Disable only the legacy first-visit reload in this isolated fixture.
 html=html.replace("if (!localStorage.getItem(flag))",'if (false)')
 (p/'index.html').write_text(html)
 (p/'probe.js').write_text("document.getElementById('root').textContent='Application started despite stalled CDN assets';")
 with sync_playwright() as pw:
  browser=pw.chromium.launch(channel='chrome',headless=True)
  page=browser.new_page()
  page.route('https://**/*',lambda route:None)
  page.goto((p/'index.html').as_uri(),wait_until='commit')
  expect(page.get_by_text('Application started despite stalled CDN assets',exact=True)).to_be_visible(timeout=5000)
  print('App entrypoint executes with fonts, sanitizer CDN and PDF CDN requests stalled')
  browser.close()
