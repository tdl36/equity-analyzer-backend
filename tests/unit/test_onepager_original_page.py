"""Synthetic original-page evidence cannot change the model's frozen context."""
import base64
import copy
import hashlib
import io
import unittest
from reportlab.pdfgen.canvas import Canvas
from onepager_corrections import prepare, preparation, original_page_text
from tests.unit.test_onepager_corrections import fixture, TEXT

class OriginalPageTests(unittest.TestCase):
 def sample(self):
  out=io.BytesIO();pdf=Canvas(out);pdf.drawString(20,700,'Old context only');pdf.showPage();pdf.drawString(20,700,TEXT);pdf.save();raw=out.getvalue()
  run,doc,body=fixture();sha=hashlib.sha256(raw).hexdigest()
  run['sources'][0].update(filename='sample.pdf',originalHash=sha,text='Old context only')
  run['input']['hashes']={'sample.pdf':sha};doc.update(filename='sample.pdf',file_data=base64.b64encode(raw).decode())
  body.update(fingerprint=preparation(run)['fingerprint'],originalPage=2)
  return run,doc,body,raw
 def test_explicit_page_only_and_report_unchanged(self):
  run,doc,body,_=self.sample();before=copy.deepcopy(run)
  _,sig,proposal=prepare(run,body,doc)
  self.assertEqual(proposal['originalPage'],2);self.assertEqual(proposal['review'],'pending');self.assertIn('outside the automated report context',proposal['caveat']);self.assertEqual(run,before)
  implicit={k:v for k,v in body.items() if k!='originalPage'}
  with self.assertRaises(ValueError):prepare(run,implicit,doc)
  self.assertEqual(prepare(run,body,doc)[1],sig)
 def test_forgery_bounds_page_and_restrictions(self):
  run,doc,body,raw=self.sample()
  for patch in [{'originalPage':1},{'originalPage':True},{'originalPage':301},{'originalPage':0},{'originalPage':3},{'excerpt':'Fabricated supporting passage not in the original.'},{'approved':True}]:
   with self.subTest(patch=patch),self.assertRaises(ValueError):prepare(run,{**body,**patch},doc)
  for bad in [{**doc,'file_data':base64.b64encode(b'changed').decode()},{**doc,'metadata':{'usage':'reference_only'}}]:
   with self.assertRaises(ValueError):prepare(run,body,bad)
  for data,name in [(raw,'sample.txt'),(b'x'*20000001,'sample.pdf')]:
   with self.assertRaises(ValueError):original_page_text(data,name,1)
 def test_edition_replays_exact_original_page_scope(self):
  from onepager_edition_contract import baseline, prepare_edition
  from tests.unit.test_onepager_edition_contract import edition_fixture, EID
  run,doc,correction,_=self.sample();_,assignment,_,_,_,financials=edition_fixture()
  ident,signature,proposal=prepare(run,correction,doc)
  body={k:v for k,v in baseline(run,assignment,financials).items() if k!='ticker'};body.update(requestId=EID,correctionIds=[ident])
  rows=[{'id':ident,'run_id':run['id'],'fingerprint':correction['fingerprint'],'payload_hash':signature,'proposal':proposal}]
  args=(run,assignment,body,rows,{'sample.pdf':doc},financials)
  self.assertEqual(prepare_edition(*args)['claims'][0]['proposal']['originalPage'],2)
  rows[0]['proposal']['originalPage']=1
  with self.assertRaises(ValueError):prepare_edition(*args)
