"""Full-app synthetic handoff at 390/1280px; serve .worker-assets on port 3012. All API calls are intercepted."""
import json,copy,re
from playwright.sync_api import sync_playwright, expect
NOTE='11111111-1111-4111-8111-111111111111'
with sync_playwright() as p:
 b=p.chromium.launch(channel='chrome',headless=True)
 for width in [390,1280]:
  page=b.new_page(viewport={'width':width,'height':1000},service_workers='block');page.add_init_script("localStorage.setItem('charlie_auth_token','synthetic')")
  revision=1;mismatch=True;writes=[];completed=False;inventory_reads=[];proposals=[];read_mode='ok';collection_mode='ok'
  def api(route):
   u=route.request.url;d={};code=200
   if route.request.method!='GET':writes.append(u)
   if '/api/auth/config' in u:d={'authRequired':True}
   elif '/api/collection/control' in u:d={'snapshot':{'policies':[],'requests':[]}}
   elif '/api/provider-credentials' in u:d={'providers':{k:{'configured':True} for k in ['anthropic','gemini','openai']}}
   elif '/api/research/assignment-defaults' in u:d={'outputs':['thesis'],'horizon':'12–24 months'}
   elif '/processing-sources' in u:
    d={'collectionId':NOTE,'ticker':'ABT','documents':[
     {'id':'thesis_1','filename':'recent.pdf','receiptSha256':'a'*64,'availability':'available','publishedDate':'2026-10-01','sourceKind':'press-release','publisher':'Company'},
     {'id':'thesis_5','filename':'older.pdf','receiptSha256':'b'*64,'availability':'available','publishedDate':'2023-07-01','sourceKind':'transcript'},
     {'id':'thesis_6','filename':'copy.pdf','receiptSha256':'a'*64,'availability':'available','publishedDate':'2026-10-01'},
     {'id':None,'filename':'missing.pdf','receiptSha256':'d'*64,'availability':'missing','publishedDate':None}]}
    if collection_mode=='mismatch':d['ticker']='CVS'
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
      assert body['plan']['labFullNote']=={'allowPdfText':True};assert body['plan']['meeting']=={'date':'2026-10-20','type':'expert'};assert body['plan']['outputs']==['meeting_prep','summary_note']
      d={'id':body['requestId'],'ticker':'ABT','collectionId':NOTE,'revision':1,'digest':'a'*64,'status':'pending','coverage':{'version':1,'total':2,'metadataRecorded':2,'collectionWindow':{'since':'2026-07-01','until':'2026-10-14'},'roles':{'collection':1,'supplemental':1,'unknown':0},'publication':{'known':2,'unknown':0,'earliest':'2024-03-01','latest':'2026-10-01','beforeWindow':1,'withinWindow':1,'afterWindow':0},'kinds':{'unknown':2},'thesisReferences':{'referenced':1,'noReference':1,'unknown':0}},'fullNote':{'modelKey':'fixture-v1','maxCalls':8,'maxTokens':32000,'allowPdfText':True,'configDigest':'c'*64,'promptSha256':'d'*64,'sourcePreparation':{'sourcePack':[{'reference':'charlie:thesis_1'},{'reference':'charlie:thesis_3'}],'batchCount':2,'representationSha256':'e'*64,'warnings':[{'reference':'charlie:thesis_3','issue':'Extracted text representation; embedded images and layout are not verified.'}]}},'sourceContext':{'1':{'role':'collection','publishedDate':'2026-10-01','referencedBySavedThesis':False},'3':{'role':'supplemental','publishedDate':'2024-03-01','referencedBySavedThesis':True}},'sources':[{'documentId':'1','reference':'charlie:thesis_1','filename':'recent.pdf','sha256':'a'*64,'bytes':10},{'documentId':'3','reference':'charlie:thesis_3','filename':'annual-2024-10-K.pdf','sha256':'b'*64,'bytes':20}],'plan':{**body['plan'],'ticker':'ABT','since':'2026-07-01','until':'2026-10-14'}};proposals.append(d)
      if width==1280:
       route.abort();return  # Saved server-side, lost response: reconcile without another submission.
    else:
     d={'proposals':copy.deepcopy(proposals)}
     if read_mode=='error':code=503;d={'error':'Saved proposal status unavailable'}
     if read_mode=='company':d['proposals'][0]['ticker']='CVS'
     if read_mode=='collection':d['proposals'][0]['collectionId']='22222222-2222-4222-8222-222222222222'
     if read_mode=='duplicate':d['proposals'].append(d['proposals'][0])
   elif '/api/research/assignments' in u:d={'assignments':[{'id':NOTE,'ticker':'ABT','status':'complete','input':{'ticker':'ABT','since':'2026-07-01','until':'2026-10-14','outputs':[],'collectionOnly':True},'result':{'sources':[{'filename':'recent.pdf','sha256':'a'*64},{'filename':'older.pdf','sha256':'b'*64},{'filename':'copy.pdf','sha256':'a'*64},{'filename':'missing.pdf','sha256':'d'*64}]}}]}
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
  # Keep synthetic layout checks independent of remote font availability.
  page.route('https://fonts.googleapis.com/**',lambda route:route.abort())
  page.route('https://fonts.gstatic.com/**',lambda route:route.abort())
  page.route('**/api/**',api)
  url='http://127.0.0.1:3012/?local=1#view=pipelinelab&ticker=ABT'
  page.goto(url,wait_until='domcontentloaded')
  page.locator('div.fixed.inset-0').filter(has=page.locator('img')).first.wait_for(state='hidden')
  page.get_by_role('button',name='Open coordinated assignment controls').click()
  page.get_by_text('Choose sources & prepare processing approval',exact=True).click()
  review=page.get_by_role('region',name='Processing approval',exact=True)
  save=review.get_by_role('button',name='Save proposal · no generation',exact=True)
  expect(save).to_be_disabled()
  assert writes==[]
  review.get_by_label('Processing purpose').select_option('event_note')
  review.get_by_role('checkbox',name='recent.pdf',exact=True).check()
  collected=review.get_by_role('region',name='Collected evidence',exact=True)
  expect(collected.get_by_role('checkbox',name='copy.pdf',exact=True)).to_be_disabled()
  expect(collected.get_by_role('checkbox',name='missing.pdf',exact=True)).to_be_disabled()
  expect(collected.get_by_role('checkbox',name='recent.pdf',exact=True).locator('xpath=../..').get_by_role('link',name='Open stored original')).to_have_attribute('href', re.compile(r'/api/pipeline-lab/original/ABT/thesis_1$'))
  collected.get_by_label('Collected source type',exact=True).select_option('Transcripts')
  expect(collected.get_by_role('checkbox',name='older.pdf',exact=True)).to_be_visible()
  expect(collected.get_by_text('1 selections hidden',exact=False)).to_be_visible()
  collected.get_by_label('Collected source type',exact=True).select_option('All')
  collected.get_by_label('Find collected sources',exact=True).fill('Company')
  expect(collected.get_by_role('checkbox',name='recent.pdf',exact=True)).to_be_checked()
  assert collected.get_by_role('checkbox',name='older.pdf',exact=True).count()==0
  collected.get_by_label('Find collected sources',exact=True).fill('')

  expect(save).to_be_disabled()
  review.get_by_label('Processing event title').fill('Guidance update')
  review.get_by_label('Processing event date').fill('2026-10-01')
  expect(save).to_be_enabled()
  collection_mode='mismatch';collected.get_by_role('button',name='Refresh collected source metadata',exact=True).click()
  expect(collected.get_by_role('alert')).to_contain_text('does not match')
  expect(save).to_be_disabled()
  expect(collected.get_by_role('checkbox',name='recent.pdf',exact=True)).to_be_checked()
  collection_mode='ok';collected.get_by_role('button',name='Refresh collected source metadata',exact=True).click()
  expect(save).to_be_enabled()
  collected.scroll_into_view_if_needed();page.screenshot(path='/tmp/charlie-T319-'+str(width)+'-sources.png')

  review.get_by_role('checkbox',name='Meeting questions',exact=True).check()
  expect(save).to_be_disabled()
  review.get_by_label('Processing meeting date').fill('2026-10-20')
  review.get_by_label('Processing meeting type').select_option('expert')
  review.get_by_role('checkbox',name='Full versioned Lab note',exact=False).check()
  review.get_by_role('checkbox',name='Allow extracted PDF text',exact=False).check()
  expect(save).to_be_enabled()
  review.get_by_role('button',name='Browse wider source library',exact=True).click()
  library=review.get_by_role('region',name='Supplemental source library',exact=True)
  expect(library.get_by_role('checkbox',name='reference-only.pdf',exact=False)).to_be_disabled()
  library.get_by_role('checkbox',name='annual-2024-10-K.pdf',exact=False).check()
  library.get_by_role('button',name='Load more library sources',exact=True).click()
  expect(library.get_by_role('checkbox',name='historical-transcript.pdf',exact=False)).to_be_visible()
  library.get_by_label('Find library sources').fill('annual')
  expect(library.get_by_role('checkbox',name='annual-2024-10-K.pdf',exact=False)).to_be_checked()
  expect(library.get_by_role('checkbox',name='annual-2024-10-K.pdf',exact=False).locator('xpath=../..').get_by_role('link',name='Open stored original')).to_have_attribute('href',re.compile(r'/api/pipeline-lab/original/ABT/thesis_3$'))
  assert not library.get_by_role('checkbox',name='historical-transcript.pdf',exact=False).count()
  page.get_by_text('Choose sources & prepare processing approval',exact=True).click()
  page.get_by_text('Choose sources & prepare processing approval',exact=True).click()
  expect(library.get_by_role('checkbox',name='annual-2024-10-K.pdf',exact=False)).to_be_checked()
  assert writes==[]
  page.reload(wait_until='domcontentloaded')
  page.locator('div.fixed.inset-0').filter(has=page.locator('img')).first.wait_for(state='hidden')
  page.get_by_role('button',name='Open coordinated assignment controls').click()
  page.get_by_text('Choose sources & prepare processing approval',exact=True).click()
  expect(review.get_by_text('Restored unsaved choices in this browser tab.',exact=False)).to_be_visible()
  expect(review.get_by_label('Processing purpose')).to_have_value('event_note')
  expect(review.get_by_label('Processing event title')).to_have_value('Guidance update')
  expect(review.get_by_label('Processing meeting type')).to_have_value('expert')
  expect(review.get_by_role('checkbox',name='Full versioned Lab note',exact=False)).to_be_checked()
  review.get_by_role('button',name='Browse wider source library',exact=True).click()
  expect(library.get_by_role('checkbox',name='annual-2024-10-K.pdf',exact=False)).to_be_checked()
  assert writes==[]
  save.click()
  if width==1280:
   expect(review.get_by_role('button',name='Retry saved proposal submission',exact=True)).to_be_visible()
   assert len(writes)==1
  else:
   approve=review.get_by_role('button',name='Approve processing · uses research credits',exact=True)
   expect(approve).to_be_visible()
   assert len(writes)==1,writes
   expect(review.get_by_role('checkbox',name='recent.pdf',exact=True)).not_to_be_checked()
   review.get_by_role('checkbox',name='recent.pdf',exact=True).check()
   review.get_by_role('button',name='Discard unsaved choices',exact=True).click()
   expect(review.get_by_role('checkbox',name='recent.pdf',exact=True)).not_to_be_checked()
   assert len(writes)==1
  page.reload(wait_until='domcontentloaded')
  page.locator('div.fixed.inset-0').filter(has=page.locator('img')).first.wait_for(state='hidden')
  page.get_by_role('button',name='Open coordinated assignment controls').click()
  page.get_by_text('Choose sources & prepare processing approval',exact=True).click()
  review=page.get_by_role('region',name='Processing approval',exact=True)
  approve=review.get_by_role('button',name='Approve processing · uses research credits',exact=True)
  expect(approve).to_be_visible()
  assert len(writes)==1
  assert not review.get_by_text('Restored unsaved choices in this browser tab.',exact=False).count()
  expect(review.get_by_text('Full versioned note · frozen model fixture-v1',exact=False)).to_be_visible()
  expect(review.get_by_text('Meeting preparation · 2026-10-20 · expert',exact=True)).to_be_visible()
  expect(review.get_by_text('Full-note source check: 2 originals planned across 2 batches.',exact=False)).to_be_visible()
  review.get_by_text('Source representation limitations · 1',exact=True).click()
  expect(review.get_by_text('Extracted text representation; embedded images and layout are not verified.',exact=False)).to_be_visible()
  expect(review.get_by_text('annual-2024-10-K.pdf: Extracted text representation;',exact=False)).to_be_visible()
  coverage=review.get_by_role('region',name='Selected evidence coverage',exact=True)
  expect(coverage.get_by_text('Recorded publication dates: 2024-03-01 – 2026-10-01.',exact=False)).to_be_visible()
  coverage.get_by_text('Source mix and thesis references',exact=True).click()
  expect(coverage.get_by_text('A reference does not prove that evidence was incorporated',exact=False)).to_be_visible()
  expect(coverage.get_by_text('1 exact-byte references found',exact=False)).to_be_visible()
  approve.scroll_into_view_if_needed()
  page.screenshot(path='/tmp/charlie-T319-'+str(width)+'-approval.png')
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'Overflow'
  for read_mode in ['error','company','collection','duplicate']:
   review.get_by_role('button',name='Refresh saved proposals',exact=True).click()
   expect(review.get_by_role('alert')).to_be_visible()
   expect(approve).to_be_disabled()
   expect(review.get_by_role('button',name='Cancel proposal',exact=True)).to_be_disabled()
   expect(review.get_by_role('button',name='Save proposal · no generation',exact=True)).to_be_disabled()
   expect(review.get_by_text('Meeting preparation · 2026-10-20 · expert',exact=True)).to_be_visible()
   assert len(writes)==1
  read_mode='ok';review.get_by_role('button',name='Refresh saved proposals',exact=True).click()
  expect(approve).to_be_enabled()
  approve.click()
  link=review.get_by_role('link',name='Open saved processing job and outputs')
  expect(link).to_have_attribute('href', '#view=stockanalysis&ticker=ABT&section=stock-analysis&work=assignment%3A'+NOTE)
  assert len(writes)==2,writes
  print(width,'supplemental library, exclusions, filtering, pagination, collapse retention, local draft reload/discard, lost-save reconciliation, saved coverage, failed/mismatched status guards and mock approval passed',flush=True)
  page.close()
 b.close()
