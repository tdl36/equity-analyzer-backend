"""Synthetic follow-up links and lost-response recovery; all API writes intercepted."""
import json
from urllib.parse import urlparse,parse_qs
from playwright.sync_api import sync_playwright,expect
WORK='33333333-3333-4333-8333-333333333333';OLD='11111111-1111-4111-8111-111111111111';ACTIVE='22222222-2222-4222-8222-222222222222'
with sync_playwright() as p:
 b=p.chromium.launch(channel='chrome',headless=True)
 for width in [390,1280]:
  page=b.new_page(viewport={'width':width,'height':1000},service_workers='block');page.add_init_script("localStorage.setItem('charlie_auth_token','synthetic')")
  writes=[];errors=[];reads=[];mode='ok';saved=None;receipt_visible=False;revision=2;submissions=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  def api(route):
   global saved,receipt_visible
   req=route.request;u=urlparse(req.url);q=parse_qs(u.query);d={};code=200
   if req.method!='GET':
    writes.append(req.url)
    if mode=='recovery' and u.path=='/api/research/amendments/ABT' and req.method=='POST':
     payload=req.post_data_json;submissions.append(payload)
     saved={'id':payload['requestId'],'ticker':'ABT','status':'queued','work_evidence':{'id':WORK,'revision':payload['workEvidence']['revision']}}
     route.abort('failed');return
    route.fulfill(status=409,json={'error':'Writes forbidden'});return
   if u.path=='/api/auth/config':d={'authRequired':True}
   elif u.path=='/api/analyses':d=[]
   elif u.path=='/api/agent/health':d={'status':'ok','staleSeconds':1}
   elif u.path=='/api/research/investment-case/ABT':d={'revision':1,'body':{},'versions':[],'bridge':{}}
   elif u.path=='/api/research/workbench/ABT':d={'records':[{'id':WORK,'revision':revision,'body':{'kind':'followup','title':'Saved management answer','status':'open','caseRevision':1,'owner':'Investor','dueDate':'2026-10-15','rationale':'Synthetic follow-up','nextAction':'Compare with thesis'}}],'sources':[],'answers':[]}
   elif u.path=='/api/research/decisions/ABT':d={'decisions':[],'revision':0,'hasMore':False}
   elif u.path=='/api/research/evidence/ABT':d={'savedThesis':{'thesis':{'summary':'Synthetic existing thesis'}},'documents':{'uploaded':[]}}
   elif u.path=='/api/research/automation':d={'enabled':False,'dailyLimit':1,'usedToday':0,'events':[],'hasServerKey':True,'manifestInitialized':True}
   elif u.path=='/api/research/amendments/ABT':
    reads.append(q)
    older={'id':OLD,'ticker':'ABT','status':'failed','error':'Saved follow-up comparison failed','work_evidence':{'id':WORK,'revision':1},'result':{}}
    active={'id':ACTIVE,'ticker':'ABT','status':'awaiting_approval'}
    if 'requestId' in q and mode=='recovery':d={'jobs':[saved] if receipt_visible and saved else []}
    elif 'requestId' in q:d={'jobs':[older] if q['requestId'][0]==OLD else [{**active,'result':{'changes':[]}}]}
    elif 'workId' in q:
     assert q['workId']==[WORK]
     d={'jobs':[older],'activeProposal':active,'truncated':True,'workId':WORK}
     if mode in ['empty','recovery']:d['jobs']=[];d['truncated']=False
     if mode=='recovery':d['activeProposal']=None
     if mode=='mismatch':d['workId']=OLD
     if mode=='error':code=503;d={'error':'Fixture unavailable'}
    else:d={'jobs':[{**active,'result':{'changes':[]}},older]}
   else:code=404
   route.fulfill(status=code,content_type='application/json',body=json.dumps(d))
  page.route('https://fonts.googleapis.com/**',lambda r:r.abort());page.route('https://fonts.gstatic.com/**',lambda r:r.abort());page.route('**/api/**',api)
  url='http://127.0.0.1:3016/?local=1#view=stockanalysis&ticker=ABT&section=reviews'
  page.goto(url,wait_until='networkidle');page.locator('div.fixed.inset-0').filter(has=page.locator('img')).first.wait_for(state='hidden')
  def handoff():
   summary=page.get_by_text('Recorded answer → thesis proposal',exact=True);summary.click()
   return page.locator('details').filter(has=page.locator(':scope > summary',has_text='Recorded answer → thesis proposal'))
  panel=handoff()
  expect(panel.get_by_text('Different saved revision',exact=False)).to_be_visible()
  expect(panel.get_by_text('Showing the 30 most recent',exact=False)).to_be_visible()
  expect(panel.get_by_role('button',name='Prepare thesis proposal from saved answer')).to_be_disabled()
  expect(panel.get_by_role('link',name='Review this saved proposal')).to_have_attribute('href',f'#view=desk&ticker=ABT&section=evidence&work=amendment%3A{OLD}')
  panel.get_by_role('link',name='Review this saved proposal').click()
  expect(page.get_by_text('Last proposal: failed',exact=False)).to_be_visible()
  page.reload(wait_until='networkidle');expect(page.get_by_text('Last proposal: failed',exact=False)).to_be_visible()
  page.go_back();panel=handoff()
  expect(panel.get_by_role('link',name='Review this saved proposal')).to_be_visible()
  page.go_forward();expect(page.get_by_text('Last proposal: failed',exact=False)).to_be_visible()
  page.go_back();panel=handoff()
  for mode in ['error','mismatch']:
   panel.get_by_role('button',name='Check proposal status').click()
   expect(panel.get_by_role('alert')).to_be_visible()
   expect(panel.get_by_role('button',name='Prepare thesis proposal from saved answer')).to_be_disabled()
   expect(panel.get_by_role('link',name='Review this saved proposal')).to_be_visible()
  mode='empty';panel.get_by_role('button',name='Check proposal status').click()
  expect(panel.get_by_text('No saved thesis proposal is linked',exact=False)).to_be_visible()
  expect(panel.get_by_role('link',name='Review the active company proposal')).to_be_visible()
  assert panel.get_by_role('link',name='Review this saved proposal').count()==0
  assert page.evaluate('document.documentElement.scrollWidth-innerWidth')<=1
  page.screenshot(path=f'/tmp/charlie-T316-{width}-followup.png')
  assert not writes and not errors,(writes,errors)
  print(json.dumps({'width':width,'linkedReview':True,'reloadBackForward':True,'emptyAndErrorDistinct':True,'writes':len(writes)}),flush=True)
  mode='recovery';panel.get_by_role('button',name='Check proposal status').click()
  panel.get_by_role('button',name='Prepare thesis proposal from saved answer').click()
  expect(panel.get_by_text('Unresolved submission',exact=False)).to_be_visible()
  assert len(submissions)==1
  original=submissions[0];receipt_visible=True;revision=3
  page.reload(wait_until='networkidle');panel=handoff()
  expect(panel.get_by_role('link',name='Review recovered proposal')).to_be_visible()
  assert len(submissions)==1,'Reload dispatched another comparison'
  expect(panel.get_by_role('link',name='Review recovered proposal')).to_have_attribute('href',f"#view=desk&ticker=ABT&section=evidence&work=amendment%3A{original['requestId']}")
  receipt_visible=False;saved=None
  panel.get_by_role('button',name='Prepare thesis proposal from saved answer').click()
  expect(panel.get_by_text('Unresolved submission',exact=False)).to_be_visible()
  frozen=submissions[-1];revision=4
  page.reload(wait_until='networkidle');panel=handoff()
  expect(panel.get_by_text('different from the current revision',exact=False)).to_be_visible()
  assert len(submissions)==2
  panel.get_by_role('button',name='Retry original saved request').click()
  expect(panel.get_by_role('button',name='Retry original saved request')).to_be_enabled()
  assert len(submissions)==3 and submissions[-1]==frozen
  receipt_visible=True;saved={**saved,'ticker':'CVS'}
  panel.get_by_role('button',name='Check proposal status').click()
  expect(panel.get_by_role('alert')).to_contain_text('does not match')
  expect(panel.get_by_role('button',name='Retry original saved request')).to_be_disabled()
  saved={**saved,'ticker':'ABT'}
  page.evaluate("()=>{window.recoveryKey=Object.keys(sessionStorage).find(k=>k.startsWith('charlie-meeting-proposal-v1:'));window.realRemove=Storage.prototype.removeItem;Storage.prototype.removeItem=function(k){if(k===window.recoveryKey)throw Error('Cleanup denied');return window.realRemove.call(this,k);};}")
  panel.get_by_role('button',name='Check proposal status').click()
  expect(panel.get_by_role('link',name='Review recovered proposal')).to_be_visible()
  expect(panel.get_by_role('alert')).to_contain_text('Cleanup denied')
  expect(panel.get_by_role('button',name='Retry original saved request')).to_be_disabled()
  assert page.evaluate('sessionStorage.getItem(window.recoveryKey)') is not None
  page.evaluate('()=>{Storage.prototype.removeItem=window.realRemove;}')
  panel.get_by_role('button',name='Check proposal status').click()
  expect(panel.get_by_role('button',name='Prepare thesis proposal from saved answer')).to_be_enabled()
  assert page.evaluate('sessionStorage.getItem(window.recoveryKey)') is None
  page.evaluate("sessionStorage.setItem(window.recoveryKey,'{')")
  mode='ok'
  page.evaluate("Object.keys(sessionStorage).filter(k=>k.startsWith('charlie-meeting-proposal-v1:')).forEach(k=>sessionStorage.setItem(k,'{'))")
  page.reload(wait_until='networkidle');panel=handoff()
  expect(panel.get_by_role('alert')).to_contain_text('unreadable')
  expect(panel.get_by_role('link',name='Review this saved proposal')).to_have_attribute('href',f'#view=desk&ticker=ABT&section=evidence&work=amendment%3A{OLD}')
  expect(panel.get_by_role('link',name='Review the active company proposal')).to_be_visible()
  expect(panel.get_by_role('button',name='Prepare thesis proposal from saved answer')).to_be_disabled()
  page.add_init_script("const originalGet=Storage.prototype.getItem;Storage.prototype.getItem=function(key){if(key.startsWith('charlie-meeting-proposal-v1:'))throw Error('Browser storage unavailable');return originalGet.call(this,key);};")
  page.reload(wait_until='networkidle');panel=handoff()
  expect(panel.get_by_role('alert')).to_contain_text('storage unavailable')
  expect(panel.get_by_role('link',name='Review this saved proposal')).to_be_visible()
  expect(panel.get_by_role('link',name='Review the active company proposal')).to_be_visible()
  expect(panel.get_by_role('button',name='Prepare thesis proposal from saved answer')).to_be_disabled()
  assert len(submissions)==3 and len(writes)==3 and not errors,(writes,errors)
  assert page.evaluate('document.documentElement.scrollWidth-innerWidth')<=1
  print(json.dumps({'width':width,'lostResponseReloadNoDispatch':True,'changedRevisionSameRequest':True,'mismatchedReceiptAndUnavailableStorageBlocked':True,'savedLinksSurviveDamagedStorage':True,'cleanupFailureRecovery':True,'mockPosts':len(submissions)}),flush=True)
  page.close()
 b.close()
