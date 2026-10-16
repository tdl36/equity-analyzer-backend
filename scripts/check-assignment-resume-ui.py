"""Synthetic assignment retry acknowledgment checks; all requests use local fixtures."""
from pathlib import Path
import subprocess,tempfile
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
fixture=r'''
import React from '__ROOT__/node_modules/react/index.js';
import {createRoot} from '__ROOT__/node_modules/react-dom/client.js';
import {ResearchAssignment} from '__ROOT__/src/research-assignment.jsx';
window.calls=[];window.failedRead=false;
window.rows=['11111111-1111-4111-8111-111111111111','22222222-2222-4222-8222-222222222222'].map((id,i)=>({id,ticker:'CVS',status:'attention',error:'Synthetic uncertain outcome '+i,updated_at:'2026-10-16T09:00:00Z',resumeRevision:String(i+1).repeat(64),input:{since:'2026-08-01',until:'2026-10-14',outputs:['summary_note']},result:{}}));
window.setInterval=fn=>{window.poll=fn;return 1};window.clearInterval=()=>{};
window.fetch=async(url,options={})=>{
 window.calls.push({url,method:options.method||'GET',body:options.body?JSON.parse(options.body):null});
 let data={},status=200;
 if(url.endsWith('/resume')){status=503;data={error:'Synthetic resume response unavailable'};}
 else if(options.method&&options.method!=='GET')throw Error('Unexpected mutation');
 else if(url.includes('/assignments?')){data={assignments:window.rows};if(window.failedRead){status=503;data={error:'Synthetic status unavailable'};}}
 else if(url.endsWith('/assignment-defaults'))data={outputs:['summary_note'],horizon:'12–24 months'};
 else throw Error('Unexpected read '+url);
 return new Response(JSON.stringify(data),{status,headers:{'Content-Type':'application/json'}});
};
createRoot(document.getElementById('root')).render(<ResearchAssignment api="/fixture" ticker="CVS"/>);
'''
with tempfile.TemporaryDirectory(prefix='charlie-resume-ui-') as folder:
 p=Path(folder);(p/'fixture.jsx').write_text(fixture.replace('__ROOT__',str(ROOT)))
 subprocess.run([str(ROOT/'node_modules/.bin/esbuild'),str(p/'fixture.jsx'),'--bundle','--outfile='+str(p/'fixture.js'),'--loader:.js=jsx'],check=True,capture_output=True)
 (p/'index.html').write_text('<meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px sans-serif;margin:16px}fieldset{min-width:0}button{margin:6px}p{overflow-wrap:anywhere}</style><div id="root"></div><script src="fixture.js"></script>')
 with sync_playwright() as pw:
  browser=pw.chromium.launch(channel='chrome',headless=True)
  for width in (390,1280):
   page=browser.new_page(viewport={'width':width,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.route('https://**/*',lambda route:route.abort())
   page.goto((p/'index.html').as_uri())
   boxes=page.get_by_role('checkbox',name='If a paid call is uncertain',exact=False)
   buttons=page.get_by_role('button',name='Resume assignment',exact=True)
   expect(boxes).to_have_count(2)
   boxes.nth(0).check();expect(boxes.nth(1)).not_to_be_checked()
   buttons.nth(1).click();expect(boxes.nth(1)).not_to_be_checked()
   calls=page.evaluate('window.calls.filter(c=>c.method==="POST")')
   assert len(calls)==1 and calls[0]['body']['acknowledgeRetry'] is False,calls
   expect(buttons.nth(0)).to_be_enabled();expect(boxes.nth(0)).not_to_be_checked()
   boxes.nth(0).check();buttons.nth(0).click();expect(boxes.nth(0)).not_to_be_checked()
   expect(buttons.nth(0)).to_be_enabled()
   calls=page.evaluate('window.calls.filter(c=>c.method==="POST")')
   assert len(calls)==2 and calls[1]['body']=={'acknowledgeRetry':True,'expectedRevision':'1'*64},calls
   boxes.nth(0).check()
   page.evaluate('async()=>{window.rows[0].resumeRevision="3".repeat(64);await window.poll()}')
   expect(boxes.nth(0)).not_to_be_checked()
   boxes.nth(0).check()
   page.evaluate('async()=>{window.failedRead=true;await window.poll()}')
   expect(buttons.nth(0)).to_be_disabled();expect(boxes.nth(0)).to_be_disabled()
   page.evaluate('async()=>{window.failedRead=false;await window.poll()}')
   expect(buttons.nth(0)).to_be_enabled();expect(boxes.nth(0)).not_to_be_checked()
   page.evaluate('async()=>{delete window.rows[0].resumeRevision;await window.poll()}')
   expect(buttons.nth(0)).to_be_disabled()
   assert len(page.evaluate('window.calls.filter(c=>c.method==="POST")'))==2
   assert not errors,errors
   print(width,'job-scoped acknowledgment, consumed failed attempt, changed snapshot, failed read and unavailable revision: passed',flush=True)
   page.close()
  browser.close()
