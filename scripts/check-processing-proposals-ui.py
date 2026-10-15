"""Full-app synthetic handoff at 390/1280px; serve .worker-assets on port 3012. All API calls are intercepted."""
import json
from playwright.sync_api import sync_playwright, expect
NOTE='11111111-1111-4111-8111-111111111111'
with sync_playwright() as p:
 b=p.chromium.launch(channel='chrome',headless=True)
 for width in [390,1280]:
  page=b.new_page(viewport={'width':width,'height':1000},service_workers='block');page.add_init_script("localStorage.setItem('charlie_auth_token','synthetic')")
  revision=1;mismatch=True;writes=[];completed=False;inventory_reads=[];proposals=[]
  def api(route):
   u=route.request.url;d={};code=200
   if route.request.method!='GET':writes.append(u)
   if '/api/auth/config' in u:d={'authRequired':True}
   elif '/api/collection/control' in u:d={'snapshot':{'policies':[],'requests':[]}}
   elif '/api/provider-credentials' in u:d={'providers':{k:{'configured':True} for k in ['anthropic','gemini','openai']}}
   elif '/api/research/assignment-defaults' in u:d={'outputs':['thesis'],'horizon':'12–24 months'}
   elif '/processing-library' in u:
    d={'documents':[{'id':'thesis_3','filename':'annual-2024-10-K.pdf','publishedDate':'2024-03-01','publisher':'Company','eligible':True},{'id':'thesis_4','filename':'reference-only.pdf','eligible':False}], 'nextCursor':'3'} if 'after=' not in u else {'documents':[{'id':'thesis_2','filename':'historical-transcript.pdf','publishedDate':'2023-07-01','eligible':True}], 'nextCursor':None}
   elif '/processing-proposals' in u:
    if route.request.method=='POST':
     body=route.request.post_data_json
     if u.endswith('/approve'):
      assert body=={'revision':1,'digest':'a'*64}
      proposals[0]['status']='dispatched';proposals[0]['assignmentId']=NOTE;d={'id':NOTE,'status':'queued'}
     else:
      assert body['filenames']==['recent.pdf'];assert body['supplementalIds']==['3'];assert body['plan']['researchPurpose']=={'kind':'event_note','event':{'title':'Guidance update','date':'2026-10-01'}}
      d={'id':body['requestId'],'ticker':'ABT','revision':1,'digest':'a'*64,'status':'pending','sourceContext':{'1':{'role':'collection','publishedDate':'2026-10-01','referencedBySavedThesis':False},'3':{'role':'supplemental','publishedDate':'2024-03-01','referencedBySavedThesis':True}},'sources':[{'documentId':'1','filename':'recent.pdf','bytes':10},{'documentId':'3','filename':'annual-2024-10-K.pdf','bytes':20}],'plan':{**body['plan'],'since':'2026-07-01','until':'2026-10-14'}};proposals.append(d)
    else:d={'proposals':proposals}
   elif '/api/research/assignments' in u:d={'assignments':[{'id':NOTE,'ticker':'ABT','status':'complete','input':{'ticker':'ABT','since':'2026-07-01','until':'2026-10-14','outputs':[],'collectionOnly':True},'result':{'sources':[{'filename':'recent.pdf','sha256':'a'*64},{'filename':'older.pdf','sha256':'b'*64}]}}]}
   elif '/api/pipeline-lab/check/' in u:d={'task':'thesis','revision':1,'checkedOnly':True,'generationAvailable':False,'blockers':[],'digest':'b'*64,'verified':[{'reference':'charlie:thesis_99' if mismatch else 'charlie:thesis_1','filename':'recent.pdf','sha256':'a'*64,'bytes':12},{'reference':'charlie:thesis_2','filename':'upload.pdf','sha256':'c'*64,'bytes':15}]}
   elif '/api/pipeline/universe' in u:d={'universe':[{'ticker':'ABT','company':'Abbott','docCount':1}]}
   elif '/api/pipeline/documents/ABT' in u:inventory_reads.append(u);d={'documents':[{'id':'thesis_1','filename':'recent.pdf','publishedDate':'2026-10-01'},{'id':'thesis_2','filename':'upload.pdf','createdAt':'2026-10-13'}]}
   elif '/api/agent/local-files/ABT' in u:d={'files':[]}
   elif '/api/pipeline-lab/config/ABT' in u:d={'revision':revision,'selection':{'thesis':['charlie:thesis_1','charlie:thesis_2'],'note':[]}}
   elif '/api/pipeline-lab/note-jobs/ABT' in u:d={'jobs':[]}
   elif '/history' in u:d={'versions':[{'revision':3,'createdAt':'2026-10-14T12:00:00Z'}]}
   elif '/api/pipeline-lab/notes/'+NOTE in u:d={'noteId':NOTE,'revision':3,'ticker':'WRONG' if mismatch else 'ABT','acceptance':{'revision':3,'acceptedAt':'2026-10-14T12:00:00Z'},'title':'Exact linked note','markdown':'## Linked research\nOlder note outside recent list.','qualityIssues':['Fixture only'],'sourcePack':[],'generationEvidence':{}}
   elif '/icloud-imports/' in u:d={'imports':[{'requestId':'fixture-import','status':'complete','receipts':[{}],'input':{'task':'thesis','revision':1},'completedRevision':2}] if completed else []}
   elif '/api/analyses' in u:d=[]
   elif '/api/agent/health' in u:d={'status':'ok','staleSeconds':1}
   else:code=404
   route.fulfill(status=code,content_type='application/json',body=json.dumps(d))
  page.route('**/api/**',api)
  url='http://127.0.0.1:3012/?local=1#view=pipelinelab&ticker=ABT'
  page.goto(url,wait_until='networkidle')
  page.locator('div.fixed.inset-0').filter(has=page.locator('img')).first.wait_for(state='hidden')
  page.get_by_role('button',name='Open coordinated assignment controls').click()
  page.get_by_text('Choose sources & prepare processing approval',exact=True).click()
  review=page.get_by_role('region',name='Processing approval',exact=True)
  save=review.get_by_role('button',name='Save proposal · no generation',exact=True)
  expect(save).to_be_disabled()
  assert writes==[]
  review.get_by_label('Processing purpose').select_option('event_note')
  review.get_by_role('checkbox',name='recent.pdf',exact=True).check()
  expect(save).to_be_disabled()
  review.get_by_label('Processing event title').fill('Guidance update')
  review.get_by_label('Processing event date').fill('2026-10-01')
  expect(save).to_be_enabled()
  review.get_by_role('button',name='Browse wider source library',exact=True).click()
  library=review.get_by_role('region',name='Supplemental source library',exact=True)
  expect(library.get_by_role('checkbox',name='reference-only.pdf',exact=False)).to_be_disabled()
  library.get_by_role('checkbox',name='annual-2024-10-K.pdf',exact=False).check()
  library.get_by_role('button',name='Load more library sources',exact=True).click()
  expect(library.get_by_role('checkbox',name='historical-transcript.pdf',exact=False)).to_be_visible()
  library.get_by_label('Find library sources').fill('annual')
  expect(library.get_by_role('checkbox',name='annual-2024-10-K.pdf',exact=False)).to_be_checked()
  assert not library.get_by_role('checkbox',name='historical-transcript.pdf',exact=False).count()
  page.get_by_text('Choose sources & prepare processing approval',exact=True).click()
  page.get_by_text('Choose sources & prepare processing approval',exact=True).click()
  expect(library.get_by_role('checkbox',name='annual-2024-10-K.pdf',exact=False)).to_be_checked()
  assert writes==[]
  save.click()
  approve=review.get_by_role('button',name='Approve processing · uses research credits',exact=True)
  expect(approve).to_be_visible()
  assert len(writes)==1,writes
  page.reload(wait_until='networkidle')
  page.locator('div.fixed.inset-0').filter(has=page.locator('img')).first.wait_for(state='hidden')
  page.get_by_role('button',name='Open coordinated assignment controls').click()
  page.get_by_text('Choose sources & prepare processing approval',exact=True).click()
  review=page.get_by_role('region',name='Processing approval',exact=True)
  approve=review.get_by_role('button',name='Approve processing · uses research credits',exact=True)
  expect(approve).to_be_visible()
  assert len(writes)==1
  approve.scroll_into_view_if_needed()
  page.screenshot(path='/tmp/charlie-T309-'+str(width)+'-approval.png')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'Overflow'
  approve.click()
  link=review.get_by_role('link',name='Open saved processing job and outputs')
  expect(link).to_have_attribute('href', '#view=stockanalysis&ticker=ABT&section=stock-analysis&work=assignment%3A'+NOTE)
  assert len(writes)==2,writes
  print(width,'supplemental library, exclusions, filtering, pagination, collapse retention, frozen roles and explicit approval passed',flush=True)
  page.close()
 b.close()
