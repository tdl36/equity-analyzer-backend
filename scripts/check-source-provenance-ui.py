"""Inspect synthetic derivative source appendices; all network requests are blocked."""
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from tests.unit.test_research_source_provenance import fixture
from research_assignment_outputs import note
from research_onepager import render as onepager
from research_infographic import render as infographic

source, assignment, run, fundamentals = fixture()
pages = {'summary': note(run, 'summary_note', assignment),
         'onepager': onepager(run, assignment),
         'infographic': infographic(run, assignment, fundamentals)}
with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    for width in (390, 1280):
        for name, content in pages.items():
            page = browser.new_page(viewport={'width': width, 'height': 1000})
            page.route('**/*', lambda route: route.abort())
            page.set_content(content, wait_until='domcontentloaded')
            # Open the source appendix containing the new shared formatter.
            register = page.locator('details').filter(has=page.locator('.source-provenance')).first
            register.locator('summary').click()
            source_details = register.locator('.source-provenance').first
            expect(source_details).to_be_visible()
            expect(source_details).to_contain_text('Supplemental history; not collected')
            expect(source_details).to_contain_text('2024-03-01')
            expect(source_details).to_contain_text('Charts were not extracted.')
            assert source_details.evaluate('(el)=>getComputedStyle(el).fontFamily').startswith('Calibri')
            assert source_details.locator('p').first.evaluate('(el)=>getComputedStyle(el).color')=='rgb(0, 0, 0)'
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), name+' horizontal overflow'
            register.scroll_into_view_if_needed()
            page.screenshot(path='/tmp/charlie-T322-'+name+'-'+str(width)+'.png')
            page.close()
            print(name, width, 'synthetic appendix content, black Calibri text and viewport checks passed', flush=True)
    browser.close()
