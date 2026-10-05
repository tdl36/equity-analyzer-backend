"""Synthetic browser/SQL workflow. Uses only a new disposable PostgreSQL cluster.
No app_v3 import, production connection, provider call, or real research approval.
"""
import json
import sys
import tempfile
import threading
import subprocess
from urllib.parse import urlparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flask import Flask, send_file
from thesis_imports import create_blueprint
from werkzeug.serving import make_server
from playwright.sync_api import sync_playwright
from tests.unit.test_thesis_imports_postgres import PostgresDraftTests
from tests.unit.test_thesis_imports import sample

repo=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='charlie-thesis-browser-') as tmp:
    root=Path(tmp)
    (root/'entry.jsx').write_text(f'''import React from {json.dumps(str(repo/'node_modules/react/index.js'))};
import {{createRoot}} from {json.dumps(str(repo/'node_modules/react-dom/client.js'))};
import {{ThesisImports}} from {json.dumps(str(repo/'src/thesis-imports.jsx'))};
createRoot(document.getElementById('root')).render(<ThesisImports onNavigate={{(view,ticker)=>{{document.title=ticker+':'+view;}}}} />);''')
    subprocess.run(['npx','esbuild',str(root/'entry.jsx'),'--bundle',f'--outfile={root}/bundle.js','--define:process.env.NODE_ENV="production"'],cwd=repo,check=True,stdout=subprocess.DEVNULL)
    PostgresDraftTests.setUpClass()
    app=Flask("browser")
    app.register_blueprint(create_blueprint(PostgresDraftTests.db))
    @app.get('/')
    def index():return '<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div><script src="/bundle.js"></script>'
    @app.get('/bundle.js')
    def bundle():return send_file(root/'bundle.js')
    @app.get('/full')
    def full():return (repo/'index.html').read_text()
    @app.get('/dist/<name>')
    def asset(name):
        if name not in ('app.js','tailwind.css'):return '',404
        return send_file(repo/'dist'/name)
    server=make_server('127.0.0.1',0,app,threaded=True)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        fixture=sample('SYNTH');fixture['companyName']=fixture['analysis']['company']='Synthetic Review Company'
        fixture['analysis']['thesis']['pillars'][0]['description']='A conditional interpretation — retention must improve. ' * 12
        fixture['sourceRegister'][0]['usageStatus']='Original terms remain applicable; import does not verify source permission.'
        fixture['analysis']['thesis']['summary']='<img src=x onerror="window.injected=true"> This is visible text, not executable markup.'
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,channel='chrome')
            page=browser.new_page(viewport={'width':1440,'height':1000})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}/');page.wait_for_load_state('networkidle')
            assert page.get_by_text('No drafts yet',exact=False).is_visible()
            page.get_by_label('Preparation ticker').fill('SYNTH')
            with page.expect_download() as download:page.get_by_role('button',name='Prepare for ChatGPT').click()
            prepared=json.loads(Path(download.value.path()).read_text());assert prepared['baseline']['expectedNoExistingThesis'] is True
            fixture['baseline']=prepared['baseline']
            page.get_by_label('Import draft file').set_input_files({'name':'synthetic.json','mimeType':'application/json','buffer':json.dumps(fixture).encode()})
            page.get_by_role('heading',name='SYNTH — Synthetic Review Company').wait_for()
            assert page.get_by_role('button',name='Approve & save thesis').is_disabled()
            assert page.evaluate('window.injected') is None
            page.screenshot(path='/tmp/charlie-t123-desktop.png',full_page=False)
            page.reload();page.wait_for_load_state('networkidle')
            page.get_by_role('button',name='SYNTH · Synthetic Review Company',exact=False).click()
            page.get_by_role('heading',name='Your approval').wait_for()
            page.get_by_role('checkbox').check();page.get_by_label('Confirm ticker').fill('OTHER')
            assert page.get_by_role('button',name='Approve & save thesis').is_disabled()
            page.get_by_label('Confirm ticker').fill('SYNTH')
            page.get_by_role('button',name='Approve & save thesis').click()
            page.get_by_role('button',name='Open saved thesis').wait_for()
            assert 'Thesis approved and saved' in page.locator('body').inner_text()
            page.get_by_role('button',name='Open saved thesis').click();assert page.title()=='SYNTH:portfolio'
            # Repeat upload reopens approved receipt and cannot approve twice.
            page.get_by_label('Import draft file').set_input_files({'name':'same.json','mimeType':'application/json','buffer':json.dumps(fixture).encode()})
            page.get_by_text('This file is already in your inbox.',exact=False).wait_for()
            assert page.get_by_role('button',name='Approve & save thesis').count()==0
            # A newly prepared upgrade is reviewed against the prior live thesis.
            with page.expect_download() as download:page.get_by_role('button',name='Prepare for ChatGPT').click()
            upgrade=json.loads(Path(download.value.path()).read_text());upgrade['analysis']['thesis']['summary']='Upgraded view pending new evidence.'
            page.get_by_label('Import draft file').set_input_files({'name':'upgrade.json','mimeType':'application/json','buffer':json.dumps(upgrade).encode()})
            page.get_by_role('heading',name='Your approval').wait_for()
            assert 'before · saved at import' in page.locator('body').inner_text().lower()
            for width in (390,320):
                page.set_viewport_size({'width':width,'height':844});page.evaluate('window.scrollTo(0,0)')
                page.screenshot(path=f'/tmp/charlie-t123-mobile-{width}.png',full_page=False)
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'),str(page.evaluate('Array.from(document.querySelectorAll("*")).filter(e=>e.getBoundingClientRect().right>innerWidth).map(e=>({tag:e.tagName,cls:e.className,width:e.getBoundingClientRect().width,right:e.getBoundingClientRect().right})).slice(0,20)'))
                page.get_by_role('heading',name='Your approval').scroll_into_view_if_needed()
                page.screenshot(path=f'/tmp/charlie-t123-approval-{width}.png',full_page=False)
            # A legacy write after import must block approval, retaining the pending draft.
            with PostgresDraftTests.db(True) as (_,cur):
                cur.execute("UPDATE portfolio_analyses SET updated_at=updated_at+INTERVAL '1 second' WHERE ticker='SYNTH'")
            page.get_by_role('button',name='Refresh inbox').click()
            page.get_by_text('The live thesis has changed.',exact=False).wait_for()
            assert page.get_by_role('button',name='Approve & save thesis').is_disabled()
            page.get_by_role('button',name='Dismiss draft').click()
            page.get_by_text('Thesis upgrade · dismissed',exact=False).wait_for()
            page.get_by_label('Import draft file').set_input_files({'name':'wrong.json','mimeType':'application/json','buffer':b'{"sourceRegister":[]}'})
            page.get_by_role('alert').wait_for();assert 'structured thesis draft' in page.get_by_role('alert').inner_text()
            assert not errors,errors
            # Exercise the actual built app shell with synthetic read responses only.
            context=browser.new_context(viewport={'width':1440,'height':1000},service_workers='block')
            context.add_init_script("localStorage.setItem('charlie_auth_token','synthetic-test-token');")
            def safe_route(route):
                parsed=urlparse(route.request.url);path=parsed.path
                if path.startswith('/api/'):
                    if route.request.method!='GET':return route.fulfill(status=403,json={'error':'Synthetic shell forbids writes'})
                    if path.startswith('/api/thesis-imports'):
                        response=app.test_client().get(path+('?' + parsed.query if parsed.query else ''))
                        return route.fulfill(status=response.status_code,json=response.json)
                    if path in ('/api/analyses','/api/overviews','/api/summaries','/api/analyst-activities','/api/research-documents'):
                        return route.fulfill(json=[])
                    return route.fulfill(json=[])
                if path=='/version':return route.fulfill(json={'version':'2026-10-05T123'})
                if parsed.hostname in ('127.0.0.1','localhost'):return route.continue_()
                return route.abort()
            context.route('**/*',safe_route)
            shell=context.new_page();shell_errors=[];shell.on('pageerror',lambda e:shell_errors.append(str(e)))
            shell.goto(f'http://127.0.0.1:{server.server_port}/full?local=1#view=thesisimports&ticker=SYNTH')
            shell.get_by_role('heading',name='Bring your research into Charlie').wait_for()
            shell.wait_for_load_state('networkidle')
            shell.locator('[class*="z-[100]"]').wait_for(state='hidden')
            assert shell.get_by_role('heading',name='Bring your research into Charlie').is_visible()
            for width in (1440,390,320):
                shell.set_viewport_size({'width':width,'height':1000 if width==1440 else 844})
                shell.screenshot(path=f'/tmp/charlie-t123-shell-{width}.png')
                assert shell.evaluate('document.documentElement.scrollWidth<=innerWidth'),f'shell overflow {width}'
            assert 'Calibri' in shell.get_by_role('heading',name='Bring your research into Charlie').evaluate('(e)=>getComputedStyle(e).fontFamily')
            assert not shell_errors,shell_errors
            context.close();browser.close()
        print('PASS: synthetic preparation/import/reload/approval/retry/upgrade/conflict/dismissal/error; 1440/390/320px; no JS errors.')
    finally:
        server.shutdown();thread.join(timeout=5);PostgresDraftTests.tearDownClass()
