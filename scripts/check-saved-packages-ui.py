"""Read-only synthetic package discovery/navigation at mobile and desktop widths."""
from pathlib import Path
import subprocess,tempfile
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
fixture=r'''
import React from '__ROOT__/node_modules/react/index.js';
import {createRoot} from '__ROOT__/node_modules/react-dom/client.js';
import {AssignmentHistory} from '__ROOT__/src/assignment-history.jsx';
window.calls=[];window.fail=false;
window.fetch=async(url,options={})=>{
 if(options.method&&options.method!=='GET')throw Error('Navigation must not mutate');
 window.calls.push(url);let q=new URL(url,'https://fixture').searchParams,t=q.get('ticker');
 return new Response(JSON.stringify(window.fail?{error:'Unavailable'}:{readOnly:true,assignments:[{id:q.has('cursor')?'22222222-2222-4222-8222-222222222222':'11111111-1111-4111-8111-111111111111',ticker:t,status:'complete',created_at:'2026-10-11T00:00:00Z',updated_at:'2026-10-13T00:00:00Z'}],nextCursor:q.has('cursor')?null:'older'}),{status:window.fail?503:200});
};
function App(){let[t,setT]=React.useState('ABT');return <><button onClick={()=>setT('CVS')}>Choose CVS</button><AssignmentHistory api="/fixture" ticker={t} expanded/></>}
createRoot(document.getElementById('root')).render(<App/>);
'''
with tempfile.TemporaryDirectory(prefix='charlie-packages-') as folder:
 p=Path(folder);(p/'fixture.jsx').write_text(fixture.replace('__ROOT__',str(ROOT)))
 subprocess.run([str(ROOT/'node_modules/.bin/esbuild'),str(p/'fixture.jsx'),'--bundle','--outfile='+str(p/'fixture.js'),'--loader:.js=jsx'],check=True,capture_output=True)
 (p/'index.html').write_text('<meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px sans-serif;margin:16px}article{margin:16px 0}button,a{display:inline-block;margin:8px}small{display:block}</style><div id="root"></div><script src="fixture.js"></script>')
 with sync_playwright() as pw:
  browser=pw.chromium.launch(channel='chrome',headless=True)
  for width in (390,1280):
   page=browser.new_page(viewport={'width':width,'height':900});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto((p/'index.html').as_uri())
   expect(page.get_by_role('heading',name='ABT saved research packages')).to_be_visible()
   links=page.get_by_role('link',name='Open package',exact=False);expect(links).to_have_count(1)
   links.first.click();assert 'ticker=ABT' in page.url and 'work=assignment%3A11111111' in page.url
   page.get_by_role('button',name='Load older assignments').click();expect(links).to_have_count(2)
   page.get_by_role('button',name='Choose CVS').click();expect(links).to_have_count(1);assert 'ticker=CVS' in links.first.get_attribute('href')
   page.get_by_role('button',name='Hide saved packages').click();page.evaluate('window.fail=true');page.get_by_role('button',name='Show saved packages').click()
   expect(page.get_by_role('alert')).to_be_visible();expect(links).to_have_count(0)
   page.evaluate('window.fail=false');page.get_by_role('button',name='Retry archive').click();expect(links).to_have_count(1)
   assert not errors,errors
   print(width,'visible packages, exact link, older pages, company isolation and failed-read recovery passed',flush=True)
   page.close()
  browser.close()
