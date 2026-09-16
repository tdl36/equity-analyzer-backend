"""Exercise email rendering without importing the application or using SMTP/database."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch, MagicMock

class LabEmailTests(TestCase):
 def test_lab_email_is_plain_professional_and_escaped(self):
  tree=ast.parse((Path(__file__).resolve().parents[2]/'app_v3.py').read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='email_summary_section')
  fn.decorator_list=[]
  scope={'request':SimpleNamespace(json={'email':'fixture@example.com','section':'summary_lab','title':'MMM','content':'<h2>Brief</h2><p><strong>Management</strong>: uncertain [P1]</p><script>alert(1)</script>','smtpConfig':{'gmail_user':'fixture@example.com','gmail_app_password':'mock-only'}}),'jsonify':lambda *args,**kw:args[0] if args else kw}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'email-function','exec'),scope)
  with patch('smtplib.SMTP') as smtp:
   result=scope['email_summary_section']()
   message=smtp.return_value.__enter__.return_value.send_message.call_args.args[0]
  html=message.get_payload()[1].get_payload(decode=True).decode()
  self.assertIn('font-size:11pt',html)
  self.assertIn('Calibri',html)
  self.assertNotIn('<script',html)
  self.assertNotIn('linear-gradient',html)
  self.assertIn('uncertain [P1]',html)
  self.assertTrue(result['success'])
if __name__=='__main__':main()
