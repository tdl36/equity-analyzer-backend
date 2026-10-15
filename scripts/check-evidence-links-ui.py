"""Full-app synthetic exact evidence links; serve .worker-assets on port 3014."""
import json
from urllib.parse import urlparse,parse_qs
from playwright.sync_api import sync_playwright,expect
OLD='11111111-1111-4111-8111-111111111111';NEW='22222222-2222-4222-8222-222222222222'
with sync_playwright() as p:
 browser=p.chromium.launch(channel='chrome',headless=True)
 for width in [390,1280]:
  page=browser.new_page(viewport={'width':width,'height':1000},service_workers='block')
  page.add_init_script("localStorage.setItem('charlie_auth_token','synthetic')")
  writes=[];reads=[];errors=[];mode='ok'
  page.on('pageerror',lambda e:errors.append(str(e)))
  def api(route):
   req=route.request;u=urlparse(req.url);d={};code=200
   if req.method!='GET':writes.append(req.url);route.fulfill(status=409,json={'error':'No writes permitted'});return
   if u.path=='/api/auth/config':d={'authRequired':True}
   elif u.path=='/api/analyses':d=[]
   elif u.path=='/api/agent/health':d={'status':'ok','staleSeconds':1}
   elif u.path=='/api/activity':d={'items':[{'id':'mp_jobs:'+OLD,'jobId':OLD,'ticker':'ETN','title':'Evidence proposal','workflow':'evidence_amendment','status':'failed','bucket':'attention','step':'Older saved proposal needs attention'}],'asOf':'2026-10-15T18:20:00Z'}
   elif u.path=='/api/research/evidence/ETN':d={'savedThesis':{'thesis':{'summary':'Synthetic existing thesis'}},'documents':{'uploaded':[]}}
   elif u.path=='/api/research/automation':d={'enabled':False,'dailyLimit':1,'usedToday':0,'events':[],'hasServerKey':True,'manifestInitialized':True}
   elif u.path=='/api/research/amendments/ETN':
    focus=parse_qs(u.query).get('requestId',[''])[0];reads.append(focus)
    older={'id':OLD,'ticker':'ETN','status':'failed','error':'Saved older failure; no retry authorized.','result':{}}
    newer={'id':NEW,'ticker':'ETN','status':'awaiting_approval','result':{'changes':[]}}
    if not focus:d={'jobs':[newer,older]}
    elif mode=='missing':d={'jobs':[]}
    elif mode=='error':code=503;d={'error':'Temporary fixture outage'}
    elif mode=='mismatch':d={'jobs':[newer]}
    else:d={'jobs':[older if focus==OLD else newer]}
   else:code=404
   route.fulfill(status=code,content_type='application/json',body=json.dumps(d))
  page.route('https://fonts.googleapis.com/**',lambda r:r.abort());page.route('https://fonts.gstatic.com/**',lambda r:r.abort())
  page.route('**/api/**',api)
  page.goto('http://127.0.0.1:3014/?local=1#view=activity',wait_until='networkidle')
  page.locator('div.fixed.inset-0').filter(has=page.locator('img')).first.wait_for(state='hidden')
  page.get_by_role('button',name='Open work ↗ ETN Evidence proposal').click()
  expect(page.get_by_text('Last proposal: failed',exact=False)).to_be_visible()
  assert 'amendment%3A'+OLD in page.url and OLD in reads
  assert page.get_by_role('button',name='Dismiss proposal',exact=True).count()==0
  page.evaluate("location.hash='view=desk&ticker=ETN&section=evidence&work=amendment%3A"+NEW+"'")
  expect(page.get_by_text('Proposal ready for your review',exact=True)).to_be_visible()
  page.go_back();expect(page.get_by_text('Last proposal: failed',exact=False)).to_be_visible()
  page.go_forward();expect(page.get_by_text('Proposal ready for your review',exact=True)).to_be_visible()
  page.go_back();expect(page.get_by_text('Last proposal: failed',exact=False)).to_be_visible()
  for mode in ['missing','error','mismatch']:
   page.reload(wait_until='networkidle')
   region=page.get_by_role('region',name='Saved evidence proposal',exact=True)
   expect(region).to_be_visible()
   expect(region.get_by_role('button',name='Retry saved proposal')).to_be_visible()
   assert page.get_by_role('button',name='Dismiss proposal',exact=True).count()==0
   assert page.get_by_role('button',name='Find material changes →',exact=True).count()==0
  mode='ok';region.get_by_role('button',name='Retry saved proposal').click()
  expect(page.get_by_text('Last proposal: failed',exact=False)).to_be_visible()
  page.screenshot(path=f'/tmp/charlie-T314-{width}-proposal.png')
  assert page.evaluate('document.documentElement.scrollWidth-innerWidth')<=1
  assert not writes and not errors,(writes,errors)
  print(json.dumps({'width':width,'exactLink':True,'backForward':True,'unavailableRefused':True,'writes':len(writes),'pageErrors':errors}),flush=True)
  page.close()
 browser.close()
