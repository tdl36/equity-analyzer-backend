"""Synthetic editor interactions; no live research, paid calls or decisions."""
from pathlib import Path
import subprocess,tempfile
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
fixture=r'''
import React from '__ROOT__/node_modules/react/index.js';
import {createRoot} from '__ROOT__/node_modules/react-dom/client.js';
import {PackageEditor} from '__ROOT__/src/package-editor.jsx';
const analysis={thesis:{summary:'Initial company view',pillars:[{id:'p',title:'Devices',description:'Conditional execution',sources:[]}]},signposts:[{id:'s',metric:'Growth',target:'High single digits',timeframe:'2027'}],threats:[{id:'r',threat:'Execution',triggerPoints:'Miss'}],conclusion:'Conditional'};
const proposal={path:'/summary/one_liner',statement:'Initial brief',basis:'interpretation',sourceId:'s',excerpt:'Original evidence retained for review.',locator:'Page 1'};
const claim={correctionId:'c1',proposal};
const edition={id:'e1',stale:false,record:{claims:[claim],correctionIds:['c1'],parentId:null}};
const draft={id:'t1',ticker:'ABT',status:'pending',stale:false,fingerprint:'fp',package:{schema:'charlie.external-thesis-draft.v1',ticker:'ABT',companyName:'Abbott',baseline:{hash:'live'},analysis,sourceRegister:[],provenance:{}}};
const pack={assignmentId:'a1',runId:'r1',fingerprint:'f',fundamentalsDigest:'g',parentId:null,editions:[edition],canDecide:false};
window.calls=[];window.failDelivery=true;window.preferences={rules:[],fingerprint:'none'};
window.fetch=async(url,opts={})=>{
 const body=opts.body?JSON.parse(opts.body):null;window.calls.push({url,body});let value={},status=200;
 if(url.endsWith('/preferences')) {if(body)window.preferences={...body,fingerprint:'saved'};value=window.preferences;}
 else if(url.endsWith('/thesis-imports/t1'))value=draft;
 else if(url.endsWith('/thesis-imports'))value={id:'t2'};
 else if(url.endsWith('/corrections'))value={id:'c2'};
 else if(url.endsWith('/onepager-editions'))value=body?{id:body.requestId}:pack;
 else if(url.includes('/view?'))value={id:'e2',status:'pending',html:'preview'};
 else if(url.endsWith('/review-package')) {value={error:'Synthetic interrupted delivery'};status=503;}
 else throw Error('Unexpected route '+url);
 return new Response(JSON.stringify(value),{status});
};
createRoot(document.getElementById('root')).render(<PackageEditor api="/fixture" job={{id:'a1',ticker:'ABT',result:{reportId:'r1',reviewPackage:{requestId:'p1',thesisDraftId:'t1',editionId:'e1'}}}}/>);
'''
with tempfile.TemporaryDirectory(prefix='charlie-editor-') as folder:
 p=Path(folder);(p/'fixture.jsx').write_text(fixture.replace('__ROOT__',str(ROOT)))
 subprocess.run([str(ROOT/'node_modules/.bin/esbuild'),str(p/'fixture.jsx'),'--bundle','--outfile='+str(p/'fixture.js'),'--loader:.js=jsx'],check=True,capture_output=True)
 (p/'index.html').write_text('<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div><script src="fixture.js"></script>')
 with sync_playwright() as pw:
  browser=pw.chromium.launch(channel='chrome',headless=True)
  for width in (390,1280):
   context=browser.new_context(viewport={'width':width,'height':900});page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto((p/'index.html').as_uri());page.get_by_role('button',name='Edit & review this package').click()
   page.get_by_label('Company and investment case').fill('Investor revised company view')
   page.get_by_role('button',name='Close editor · keep edits').click();page.get_by_role('button',name='Edit & review this package').click()
   expect(page.get_by_label('Company and investment case')).to_have_value('Investor revised company view')
   page.get_by_role('button',name='One-pager',exact=True).click();page.get_by_label('summary · one liner').fill('Investor revised brief')
   beforePreview=len(page.evaluate('window.calls'))
   page.get_by_role('button',name='Preview draft before saving',exact=True).click()
   preview=page.get_by_role('region',name='Unsaved package preview')
   expect(preview.get_by_text('Investor revised company view',exact=True)).to_be_visible()
   expect(preview.get_by_text('Investor revised brief',exact=True)).to_be_visible()
   expect(preview.get_by_text('Changed in this draft',exact=True)).to_have_count(2)
   preview.scroll_into_view_if_needed()
   page.screenshot(path=f'/tmp/charlie-package-preview-{width}.png')
   assert len(page.evaluate('window.calls'))==beforePreview
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   page.get_by_role('button',name='Hide draft preview',exact=True).click()
   page.get_by_role('button',name='Writing preferences').click();page.get_by_label('Remember for future writing').fill('Do not name brokers in narrative.')
   page.get_by_role('button',name='Remember this preference').click();expect(page.get_by_role('button',name='Remove',exact=True)).to_be_visible()
   page.get_by_role('button',name='Remove',exact=True).click();expect(page.get_by_role('button',name='Remove',exact=True)).to_have_count(0)
   page.get_by_role('button',name='One-pager',exact=True).click();page.get_by_label('I checked my revised wording',exact=False).check()
   page.get_by_role('button',name='Save new package version').click();expect(page.get_by_role('alert')).to_have_text('Synthetic interrupted delivery')
   page.get_by_role('button',name='Retry saving this exact revision').click();expect(page.get_by_role('alert')).to_have_text('Synthetic interrupted delivery')
   calls=page.evaluate('window.calls');posts=[c for c in calls if c['body']]
   deliveries=[c['body'] for c in posts if c['url'].endswith('/review-package')];assert len(deliveries)==2 and deliveries[0]==deliveries[1]
   assert len([c for c in posts if c['url'].endswith('/corrections')])==1
   assert not any('/approve' in c['url'] or '/decision' in c['url'] for c in calls)
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');assert not errors,errors
   print(width,'editing, recovery, preferences/removal, interrupted save replay and approval separation passed',flush=True)
   context.close()
  browser.close()
