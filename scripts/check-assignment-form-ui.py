"""Synthetic unfinished-assignment navigation/reload checks; fixture GETs only."""
from pathlib import Path
import subprocess,tempfile
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
fixture=r'''
import React from '__ROOT__/node_modules/react/index.js';
import {createRoot} from '__ROOT__/node_modules/react-dom/client.js';
import {ResearchAssignment} from '__ROOT__/src/research-assignment.jsx';
window.calls=[];window.allowSubmit=false;window.recovery=null;
window.fetch=async(url,options={})=>{
 window.calls.push({url,method:options.method||'GET'});
 if(options.method&&options.method!=='GET'){
  if(!window.allowSubmit||!url.endsWith('/assignments'))throw Error('Unexpected submission');
  const body=JSON.parse(options.body);
  return new Response(JSON.stringify({id:body.requestId,ticker:body.ticker,status:'queued',input:body,result:{}}),{status:202,headers:{'Content-Type':'application/json'}});
 }
 const data=url.includes('requestId=')&&window.recovery?{assignments:[window.recovery]}:url.endsWith('/assignment-defaults')?{outputs:['thesis'],horizon:'Standard saved horizon'}:{assignments:[]};
 return new Response(JSON.stringify(data),{status:200,headers:{'Content-Type':'application/json'}});
};
function App(){const [ticker,setTicker]=React.useState('CVS'),[show,setShow]=React.useState(true),[revision,setRevision]=React.useState(null);
 return <><button onClick={()=>setTicker(ticker==='CVS'?'ABT':'CVS')}>Other company</button><button onClick={()=>setShow(!show)}>Toggle form</button><button onClick={()=>setRevision(r=>r===1?2:1)}>Change Lab revision</button>{show&&<ResearchAssignment api="/fixture" ticker={ticker} labMode labSelection={revision?{task:'thesis',revision,digest:'frozen'}:null}/>}</>;
}
createRoot(document.getElementById('root')).render(<App/>);
'''
with tempfile.TemporaryDirectory(prefix='charlie-assignment-form-') as folder:
 p=Path(folder);(p/'fixture.jsx').write_text(fixture.replace('__ROOT__',str(ROOT)))
 subprocess.run([str(ROOT/'node_modules/.bin/esbuild'),str(p/'fixture.jsx'),'--bundle','--outfile='+str(p/'fixture.js'),'--loader:.js=jsx'],check=True,capture_output=True)
 (p/'index.html').write_text('<meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px sans-serif;margin:16px}input,textarea,select{max-width:100%;box-sizing:border-box}fieldset{min-width:0}button{margin:6px}p{overflow-wrap:anywhere}</style><div id="root"></div><script src="fixture.js"></script>')
 with sync_playwright() as pw:
  browser=pw.chromium.launch(channel='chrome',headless=True)
  for width in (390,1280):
   page=browser.new_page(viewport={'width':width,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.route('https://**/*',lambda route:route.abort())
   page.goto((p/'index.html').as_uri())
   purpose=page.get_by_role('combobox',name='Research purpose',exact=True)
   expect(purpose).to_be_enabled()
   purpose.select_option('event_note')
   page.get_by_role('textbox',name='Event title',exact=True).fill('Guidance update')
   page.get_by_label('Event date',exact=True).fill('2026-10-15')
   page.get_by_label('Research from',exact=True).fill('2026-08-01')
   page.get_by_label('Research through',exact=True).fill('2026-10-15')
   page.get_by_role('spinbutton',name='Collection scope',exact=True).fill('200')
   page.get_by_role('checkbox',name='Collect sources only',exact=False).check()
   page.get_by_text('Research focus and defaults',exact=True).click()
   focus=page.get_by_role('textbox',name='Focus or questions (optional)',exact=True)
   focus.fill('Preserve contrary evidence and contextual history')
   page.get_by_role('button',name='Other company',exact=True).click()
   expect(purpose).to_have_value('thesis_review')
   expect(page.get_by_role('spinbutton',name='Collection scope',exact=True)).to_have_value('40')
   page.get_by_role('button',name='Other company',exact=True).click()
   for action in ('return','toggle','reload'):
    if action=='toggle':
     page.get_by_role('button',name='Toggle form',exact=True).click();page.get_by_role('button',name='Toggle form',exact=True).click()
    if action=='reload':page.reload()
    expect(page.get_by_text('Restored unfinished research choices.',exact=False)).to_be_visible()
    expect(purpose).to_have_value('event_note')
    expect(page.get_by_role('textbox',name='Event title',exact=True)).to_have_value('Guidance update')
    expect(page.get_by_label('Research from',exact=True)).to_have_value('2026-08-01')
    expect(page.get_by_role('spinbutton',name='Collection scope',exact=True)).to_have_value('200')
    expect(page.get_by_role('checkbox',name='Collect sources only',exact=False)).to_be_checked()
    page.get_by_text('Research focus and defaults',exact=True).click()
    expect(focus).to_have_value('Preserve contrary evidence and contextual history')
   page.get_by_role('button',name='Change Lab revision',exact=True).click()
   expect(purpose).to_have_value('thesis_review')
   page.get_by_text('Research focus and defaults',exact=True).click();focus.fill('Revision one only')
   page.get_by_role('button',name='Change Lab revision',exact=True).click()
   page.get_by_text('Research focus and defaults',exact=True).click();expect(focus).to_have_value('')
   page.get_by_role('button',name='Change Lab revision',exact=True).click()
   page.get_by_text('Research focus and defaults',exact=True).click();expect(focus).to_have_value('Revision one only')
   assert all(c['method']=='GET' for c in page.evaluate('window.calls'))
   page.evaluate('''()=>{
    const body={requestId:'33333333-3333-4333-8333-333333333333',ticker:'CVS',since:'2026-08-01',until:'2026-10-15',outputs:['thesis'],instruction:'Another source version',horizon:'12 months',labSelection:{task:'thesis',revision:999,digest:'other'}};
    sessionStorage.setItem('charlie-assignment-submit-v1:%2Ffixture:CVS',JSON.stringify({version:1,body}));
    window.recovery={id:body.requestId,ticker:'CVS',input:body,result:{},status:'queued'};
   }''')
   page.get_by_role('button',name='Toggle form',exact=True).click();page.get_by_role('button',name='Toggle form',exact=True).click()
   page.get_by_role('button',name='Check whether assignment was saved',exact=True).click()
   expect(page.get_by_text('Saved assignment found (queued).',exact=False)).to_be_visible()
   assert page.evaluate('Object.keys(sessionStorage).filter(k=>k.startsWith("charlie-assignment-form-v1:")).some(k=>JSON.parse(sessionStorage.getItem(k)).form.instruction==="Revision one only")')
   page.evaluate('()=>{for(const k of Object.keys(sessionStorage))if(k.startsWith("charlie-assignment-form-v1:"))sessionStorage.setItem(k,"corrupt form")}')
   page.reload()
   expect(page.get_by_text('Browser form recovery is unreadable.',exact=False)).to_be_visible()
   expect(purpose).to_be_enabled()
   assert page.evaluate('Object.keys(sessionStorage).filter(k=>k.startsWith("charlie-assignment-form-v1:")).every(k=>sessionStorage.getItem(k)==="corrupt form")')
   page.get_by_role('button',name='Clear browser form recovery',exact=True).click()
   page.get_by_text('Research focus and defaults',exact=True).click()
   page.evaluate('()=>{window.storageSet=Storage.prototype.setItem;Storage.prototype.setItem=function(k,v){if(!k.startsWith("charlie-assignment-form-v1:"))window.storageSet.call(this,k,v)}}')
   focus.fill('New recovery')
   expect(page.get_by_text('Browser form recovery could not be verified.',exact=False)).to_be_visible()
   page.evaluate('()=>{Storage.prototype.setItem=window.storageSet}')
   page.get_by_role('button',name='Clear browser form recovery',exact=True).click()
   focus.fill('Reviewed synthetic submission')
   assert all(c['method']=='GET' for c in page.evaluate('window.calls'))
   page.evaluate('()=>{window.allowSubmit=true}')
   page.get_by_role('button',name='Start research for CVS',exact=True).click()
   expect(page.get_by_text('Assignment saved. Charlie will collect originals',exact=False)).to_be_visible()
   assert len(page.evaluate('window.calls.filter(c=>c.method==="POST")'))==1
   # The successful fixture receipt clears only this form and its exact request.
   assert page.evaluate('Object.keys(sessionStorage).filter(k=>k.startsWith("charlie-assignment-form-v1:")).every(k=>sessionStorage.getItem(k)==="corrupt form")')
   page.reload()
   expect(purpose).to_be_enabled()
   expect(page.get_by_text('Restored unfinished research choices.',exact=False)).to_have_count(0)
   assert all(c['method']=='GET' for c in page.evaluate('window.calls'))
   assert not errors,errors
   print(width,'navigation/reload, company/source-version isolation, corrupt and silent-write recovery, explicit fixture submission cleanup: passed',flush=True)
   page.close()
  browser.close()
