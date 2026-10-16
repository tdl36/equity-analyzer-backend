"""Synthetic CollectionCloud navigation/read-failure checks; no network or real commands."""
from pathlib import Path
import subprocess,tempfile
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
fixture=r'''
import React from '__ROOT__/node_modules/react/index.js';
import {createRoot} from '__ROOT__/node_modules/react-dom/client.js';
import {CollectionCloud} from '__ROOT__/src/collection-cloud.jsx';
const ids={CVS:'11111111-1111-4111-8111-111111111111',ABT:'22222222-2222-4222-8222-222222222222'};
window.calls=[];window.readMode='ok';
window.fetch=async(url,options={})=>{
 window.calls.push({url,method:options.method||'GET'});
 if(options.method&&options.method!=='GET')throw Error('Unexpected command');
 let data={},status=200;
 if(url.includes('/api/collection/control')){
  if(window.readMode==='failed'){status=503;data={error:'Synthetic collection status unavailable'};}
  else data={updatedAt:window.readMode==='stale'?'2020-01-01T00:00:00Z':new Date().toISOString(),snapshot:{policies:[],requests:Object.keys(ids).map(ticker=>({id:ids[ticker],ticker,status:'attention',issue:'Synthetic '+ticker+' blocker'})),folders:{}},commands:[]};
 }else if(url.includes('/api/collection/synthesis'))data={paused:true,rows:[]};
 else data={requests:[],policies:[],events:[],paused:true,config:{enabled:false,automatic:false,tickers:[],dailyLimit:3}};
 return new Response(JSON.stringify(data),{status,headers:{'Content-Type':'application/json'}});
};
function App(){
 const [ticker,setTicker]=React.useState('CVS'),[work,setWork]=React.useState('collection:'+ids.CVS);
 const open=t=>{setTicker(t);setWork('collection:'+ids[t]);};
 return <><nav><button onClick={()=>open('CVS')}>Fixture CVS link</button><button onClick={()=>open('ABT')}>Fixture ABT link</button></nav><CollectionCloud api="/fixture" initialTicker={ticker} routeWork={work} onWorkChange={setWork}/></>;
}
createRoot(document.getElementById('root')).render(<App/>);
'''
with tempfile.TemporaryDirectory(prefix='charlie-collection-ui-') as folder:
 p=Path(folder);(p/'fixture.jsx').write_text(fixture.replace('__ROOT__',str(ROOT)))
 subprocess.run([str(ROOT/'node_modules/.bin/esbuild'),str(p/'fixture.jsx'),'--bundle','--outfile='+str(p/'fixture.js'),'--loader:.js=jsx'],check=True,capture_output=True)
 (p/'index.html').write_text('<meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px sans-serif;margin:16px}input,textarea,select{max-width:100%;box-sizing:border-box}fieldset{min-width:0}button{margin:6px}p{overflow-wrap:anywhere}</style><div id="root"></div><script src="fixture.js"></script>')
 with sync_playwright() as pw:
  browser=pw.chromium.launch(channel='chrome',headless=True)
  for width in (390,1280):
   page=browser.new_page(viewport={'width':width,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.route('https://**/*',lambda route:route.abort())
   page.goto((p/'index.html').as_uri())
   requests=page.get_by_role('region',name='Collection requests',exact=True)
   expect(requests.get_by_role('heading',name='CVS collection',exact=True)).to_be_visible()
   expect(requests.get_by_text('Synthetic CVS blocker',exact=True)).to_be_visible()
   page.get_by_text('Create or edit a collection policy',exact=True).click()
   page.get_by_role('textbox',name='Collection instructions',exact=True).fill('Keep my unfinished policy edits')
   for ticker in ('ABT','CVS','ABT'):
    page.get_by_role('button',name='Fixture '+ticker+' link',exact=True).click()
    expect(requests.get_by_role('textbox',name='Find collection company',exact=True)).to_have_value(ticker)
    expect(requests.get_by_text('Synthetic '+ticker+' blocker',exact=True)).to_be_visible()
    other='CVS' if ticker=='ABT' else 'ABT'
    expect(requests.get_by_text('Synthetic '+other+' blocker',exact=True)).to_have_count(0)
    expect(page.get_by_role('textbox',name='Collection instructions',exact=True)).to_have_value('Keep my unfinished policy edits')
   requests.get_by_role('button',name='Show all company requests',exact=True).click()
   requests.get_by_role('textbox',name='Find collection company',exact=True).fill('CVS')
   expect(requests.get_by_text('Synthetic CVS blocker',exact=True)).to_be_visible()
   page.get_by_role('button',name='Fixture ABT link',exact=True).click()
   expect(requests.get_by_role('textbox',name='Find collection company',exact=True)).to_have_value('ABT')
   retry=requests.get_by_role('button',name='Retry after resolving',exact=True)
   cancel=requests.get_by_role('button',name='Cancel this refresh',exact=True)
   expect(retry).to_be_enabled();expect(cancel).to_be_enabled()
   for mode in ('failed','stale'):
    page.evaluate('(mode)=>{window.readMode=mode}',mode)
    page.get_by_role('button',name='Check status',exact=True).click()
    expect(page.get_by_text('Collection request actions are paused',exact=False)).to_be_visible()
    expect(requests.get_by_text('Synthetic ABT blocker',exact=True)).to_be_visible()
    expect(retry).to_be_disabled();expect(cancel).to_be_disabled()
   page.evaluate('()=>{window.readMode="ok"}')
   page.get_by_role('button',name='Check status',exact=True).click()
   expect(retry).to_be_enabled();expect(cancel).to_be_enabled()
   expect(page.get_by_role('textbox',name='Collection instructions',exact=True)).to_have_value('Keep my unfinished policy edits')
   assert not errors,errors
   assert all(c['method']=='GET' for c in page.evaluate('window.calls'))
   print(width,'company/request navigation, manual filters, preserved policy edits, failed/stale reads and recovered controls: passed',flush=True)
   page.close()
  browser.close()
