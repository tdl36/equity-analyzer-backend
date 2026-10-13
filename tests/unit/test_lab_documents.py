import io,json,unittest,zipfile
from contextlib import contextmanager
from unittest.mock import patch
from flask import Flask
from PIL import Image
from PyPDF2 import PdfWriter,PdfReader
import lab_documents as docs
import summary_lab as lab


def pdf(n=6):
    w=PdfWriter()
    for _ in range(n):w.add_blank_page(width=300,height=300)
    b=io.BytesIO();w.write(b);return b.getvalue()

def png():
    b=io.BytesIO();Image.new('RGB',(30,30),'white').save(b,format='PNG');return b.getvalue()

class Documents(unittest.TestCase):
    def test_pdf_keeps_every_page_even_without_searchable_text(self):
        import base64
        d=docs.prepare('scan.pdf',pdf())
        self.assertEqual(d['pages'],6);self.assertEqual(len(d['parts']),2)
        self.assertEqual(sum(len(PdfReader(io.BytesIO(base64.b64decode(p['blocks'][0]['source']['data']))).pages) for p in d['parts']),6)
        self.assertTrue(all(p['visual'] for p in d['parts']))
        self.assertNotIn('parts',docs.manifest(d))
    def test_encrypted_empty_oversize_and_unsupported_are_rejected(self):
        w=PdfWriter();w.add_blank_page(width=100,height=100);w.encrypt('password');b=io.BytesIO();w.write(b)
        for name,data in [('locked.pdf',b.getvalue()),('empty.txt',b''),('script.exe',b'abc'),('large.txt',b'x'*(docs.MAX_BYTES+1))]:
            with self.assertRaises(ValueError):docs.prepare(name,data)
    def test_office_embedded_image_and_text_both_reach_parts(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:
            z.writestr('ppt/slides/slide1.xml','<a><t>Revenue rose.</t></a>');z.writestr('ppt/media/image1.png',png())
        d=docs.prepare('deck.pptx',b.getvalue())
        self.assertTrue(any(p['visual'] for p in d['parts']));self.assertTrue(any('Revenue' in p['text'] for p in d['parts']))
        self.assertIn('not rendered',d['limitations'][0])
    def test_workbook_retains_cells_and_formula_limitations(self):
        from openpyxl import Workbook
        w=Workbook();w.active['A1']='Revenue';w.active['B1']='=1+2';b=io.BytesIO();w.save(b)
        d=docs.prepare('model.xlsx',b.getvalue());text='\n'.join(p['text'] for p in d['parts'])
        self.assertIn('A1: Revenue',text);self.assertIn('B1: =1+2',text);self.assertIn('not recalculated',text)
    def test_every_review_and_section_check_receives_original_visual(self):
        d=docs.prepare('figure.png',png());seen=[]
        def ask(system,prompt,tokens):
            seen.append(prompt)
            if isinstance(prompt,list):
                if 'Return JSON' in prompt[-1]['text']:return json.dumps({'record':'The image has no legible data.','passages':[],'issues':['No numbers discernible.']})
                return 'No definite issues.'
            return 'Synthetic section [P1].'
        state={};lab.generate('',state,ask,lambda _:None,documents=[d])
        visual=[p for p in seen if isinstance(p,list)]
        self.assertEqual(len(visual),1+len(lab.sections_for_mode('english')))
        self.assertTrue(all(any(b['type']=='image' for b in p) for p in visual))
        self.assertEqual(state['sourceParts'][0]['label'],'figure.png')
        before=len(seen);lab.generate('',state,ask,lambda _:None,documents=[d]);self.assertEqual(before,len(seen))
        self.assertEqual(state['parts']['0']['passages'],[])
    def test_upload_preserves_original_and_replayed_start_does_not_dispatch_twice(self):
        rows={};runs={};out=None
        class Cursor:
            def execute(self,sql,args=()):
                nonlocal out
                out=None
                if sql.startswith('INSERT INTO summary_lab_documents'):
                    rows[args[0]]=dict(id=args[0],filename=args[1],original=args[2],sha256=args[3],metadata=json.loads(args[4]))
                elif sql.startswith('SELECT * FROM summary_lab_documents'):out=rows.get(args[0])
                elif sql.startswith('SELECT filename,original'):out=rows.get(args[0])
                elif sql.startswith('SELECT source_hash'):out=runs.get(args[0])
                elif sql.startswith('INSERT INTO summary_lab_experiments('):runs[args[0]]={'source_hash':args[3]}
            def fetchone(self):return out
        @contextmanager
        def db(**kwargs):yield None,Cursor()
        app=Flask(__name__);app.register_blueprint(lab.create_blueprint(db));client=app.test_client()
        with patch('provider_credentials.get',return_value='synthetic'),patch.object(lab.threading,'Thread') as thread:
            raw=pdf(1);r=client.post('/api/summary-lab/documents',data={'file':(io.BytesIO(raw),'original.pdf')})
            self.assertEqual(r.status_code,200,r.json);ident=r.json['id'];thread.assert_not_called()
            self.assertEqual(client.get('/api/summary-lab/documents/'+ident+'/original').data,raw)
            self.assertEqual(client.post('/api/summary-lab',json={'documentIds':[ident]}).status_code,400)
            thread.assert_not_called()
            payload={'documentIds':[ident],'documentCoverageAccepted':True,'requestId':'00000000-0000-4000-8000-000000000099','source':'','title':'Fixture'}
            self.assertEqual(client.post('/api/summary-lab',json=payload).status_code,202)
            self.assertEqual(client.post('/api/summary-lab',json=payload).status_code,200)
            thread.assert_called_once()
            payload['title']='Changed';self.assertEqual(client.post('/api/summary-lab',json=payload).status_code,409)
