"""Synthetic rewrite UX: no production API or model calls."""
from pathlib import Path
import subprocess,tempfile
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
fixture=r'''
import React from '__ROOT__/node_modules/react/index.js';
import {createRoot} from '__ROOT__/node_modules/react-dom/client.js';
import {InvestmentCase} from '__ROOT__/src/investment-case.jsx';
import {createApiReadScheduler} from '__ROOT__/src/api-read-scheduler.mjs';
const schedule=createApiReadScheduler();window.calls=[];window.polls=[];const nativeInterval=window.setInterval;window.setInterval=(fn,ms)=>{if(ms===6000){window.polls.push(fn);return 123;}return nativeInterval(fn,ms);};
for(let i=0;i<8;i++)schedule('/api/background',()=>new Promise(()=>{}));
window.fetch=(url,opts={})=>schedule(url,async()=>{window.calls.push(url);let value={};
 if(url.includes('/investment-case/'))value={revision:0,body:{},versions:[],bridge:{}};
 else if(url.includes('/assignment-history'))value={readOnly:true,assignments:[],nextCursor:null};
 else if(url.includes('/assignments?'))value={assignments:[{id:'00000000-0000-4000-8000-000000000001',ticker:'ABT',status:'complete',input:{since:'2026-01-01',until:'2026-10-01',outputs:[]},result:{}}]};
 else if(url.includes('/stock-analysis/'))value={runs:[],documents:[]};
 return new Response(JSON.stringify(value));},opts.signal);
createRoot(document.getElementById('root')).render(<InvestmentCase api="" initialTicker="ABT" initialTab="stock-analysis" routeSection="stock-analysis" routeWork={location.hash.includes('focused')?'assignment:00000000-0000-4000-8000-000000000001':''}/>);
'''
with tempfile.TemporaryDirectory(prefix='charlie-rewrite-') as folder:
 p=Path(folder);(p/'fixture.jsx').write_text(fixture.replace('__ROOT__',str(ROOT)))
 subprocess.run([str(ROOT/'node_modules/.bin/esbuild'),str(p/'fixture.jsx'),'--bundle','--outfile='+str(p/'fixture.js')],check=True,capture_output=True)
 (p/'index.html').write_text('<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div><script src="fixture.js"></script>')
 with sync_playwright() as pw:
  browser=pw.chromium.launch(channel='chrome',headless=True)
  for width in (390,1280):
   context=browser.new_context(viewport={'width':width,'height':900});page=context.new_page();page.goto((p/'index.html').as_uri())
   expect(page.get_by_role('heading',name='ABT saved research packages')).to_be_visible()
   calls=page.evaluate('window.calls');assert not any('underweight' in c or 'source-proposals' in c for c in calls),calls
   page.goto((p/'index.html').as_uri()+'#focused')
   page.reload()
   expect(page.get_by_role('heading',name='ABT · Research review')).to_be_visible()
   expect(page.get_by_text('Sources, progress & history',exact=True)).to_be_visible()
   before=page.evaluate('window.calls.length');page.evaluate('window.polls.forEach(fn=>fn())');assert page.evaluate('window.calls.length')==before
   calls=page.evaluate('window.calls');assert not any('/stock-analysis/' in c or '/collection/control' in c or 'assignment-defaults' in c for c in calls),calls
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   print(width,'ordinary company and focused review open with background capacity saturated; hidden reads absent')
   context.close()
  browser.close()
