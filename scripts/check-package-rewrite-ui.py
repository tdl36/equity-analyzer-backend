"""Synthetic rewrite UX: no production API or model calls."""
from pathlib import Path
import subprocess,tempfile
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
fixture=r'''
import React from '__ROOT__/node_modules/react/index.js';
import {createRoot} from '__ROOT__/node_modules/react-dom/client.js';
import {PackageRewrite} from '__ROOT__/src/package-rewrite.jsx';
window.calls=[];window.fail=true;window.kept=null;
window.fetch=async(url,opts={})=>{const body=opts.body?JSON.parse(opts.body):null;window.calls.push({url,body});let value={};
 if(url.endsWith('/prepare'))value={fingerprint:'f',estimatedUsd:0.1,fieldCount:1,sourceCount:2,calls:1,model:'synthetic',notice:'Synthetic estimate'};
 else if(url.endsWith('/start')){if(window.fail){window.fail=false;throw Error('Synthetic connection interrupted');}value={id:body.requestId};}
 else value={status:'complete',state:{rewrite:{summary:'Alternative wording'}}};
 return new Response(JSON.stringify(value));};
function Fixture(){const [changed,setChanged]=React.useState(false);return <><button onClick={()=>setChanged(!changed)}>Change related thesis</button><PackageRewrite runId="r1" api="/fixture" storageKey="synthetic" fields={[{id:'summary',label:'Company view',text:'Original wording',limit:300},{id:'case',label:'Related thesis',text:changed?'Changed case':'Original case',limit:40000}]} onKeep={x=>{window.kept=x;}}/></>;}
createRoot(document.getElementById('root')).render(<Fixture/>);
'''
with tempfile.TemporaryDirectory(prefix='charlie-rewrite-') as folder:
 p=Path(folder);(p/'fixture.jsx').write_text(fixture.replace('__ROOT__',str(ROOT)))
 subprocess.run([str(ROOT/'node_modules/.bin/esbuild'),str(p/'fixture.jsx'),'--bundle','--outfile='+str(p/'fixture.js')],check=True,capture_output=True)
 (p/'index.html').write_text('<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div><script src="fixture.js"></script>')
 with sync_playwright() as pw:
  browser=pw.chromium.launch(channel='chrome',headless=True)
  for width in (390,1280):
   context=browser.new_context(viewport={'width':width,'height':900});page=context.new_page();page.goto((p/'index.html').as_uri())
   page.get_by_role('button',name='Revise with AI',exact=True).click();page.get_by_role('combobox').select_option('summary');page.get_by_label('Instructions',exact=True).fill('Make specific')
   page.get_by_role('button',name='Review scope & cost').click();expect(page.get_by_text('Synthetic estimate',exact=True)).to_be_visible()
   assert page.evaluate('window.calls')[0]['body']['contextFields'][0]['id']=='case'
   assert not any(c['url'].endswith('/start') for c in page.evaluate('window.calls'))
   page.get_by_role('button',name='Authorize cost & generate alternative').click();expect(page.get_by_role('alert')).to_have_text('Synthetic connection interrupted')
   first=page.evaluate('window.calls').copy();page.reload()
   page.get_by_role('button',name='Check saved result').click();expect(page.get_by_text('Alternative wording',exact=True)).to_be_visible()
   assert not any(c['body'] for c in page.evaluate('window.calls'))
   assert page.evaluate('window.kept') is None
   page.get_by_role('button',name='Change related thesis',exact=True).click()
   page.get_by_role('button',name='Keep alternative in my draft').click()
   expect(page.get_by_role('alert')).to_contain_text('package wording or context changed')
   assert page.evaluate('window.kept') is None
   page.get_by_role('button',name='Change related thesis',exact=True).click()
   page.get_by_role('button',name='Keep alternative in my draft').click()
   assert page.evaluate('window.kept')=={'summary':'Alternative wording'}
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   print(width,'explicit estimate/authorization, interrupted reload GET recovery, comparison and deliberate keep passed')
   context.close()
  browser.close()
