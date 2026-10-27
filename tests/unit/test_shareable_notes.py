import ast
from pathlib import Path
import unittest
import shareable_notes
import summary_lab_editorial

class ShareableNotesTests(unittest.TestCase):
    def test_known_roles_and_separate_issues_are_preserved(self):
        policy=shareable_notes.RULES
        for phrase in ['CEO, CFO, IR','Never guess','Do not rename, suppress or sanitize the Transcript Corrections Log','Key Unresolved Items / Next Checks','raw transcript/source itself is preserved verbatim','Do not remove genuine business uncertainty']:
            self.assertIn(phrase,policy)
        self.assertIn(policy,summary_lab_editorial.RULES)
        self.assertIn('else RULES + shareable_notes.RULES',Path('summary_lab.py').read_text())

    def test_placeholder_diagnostics_distinguish_real_roles(self):
        for text in ['Speaker 1: growth improves','Lead exec: outlook unchanged','Unidentified speaker: yes','garbled transcript']:
            self.assertTrue(summary_lab_editorial.review_flags(text),text)
        self.assertEqual(summary_lab_editorial.review_flags('CFO: margins could improve. CEO expects growth.'),[])

    def test_all_backend_summary_writers_receive_shared_contract(self):
        tree=ast.parse(Path('app_v3.py').read_text())
        names={'_run_podcast_fullsummary_job','_meeting_notes_html','generate_summary','generate_assessment','_run_auto_process_text','_run_auto_process_audio'}
        covered=set();count=0
        for f in tree.body:
            if isinstance(f,ast.FunctionDef) and f.name in names:
                for call in ast.walk(f):
                    if isinstance(call,ast.Call):
                        for kw in call.keywords:
                            if kw.arg in ('system','system_instruction'):
                                self.assertIn('shareable_notes.RULES',ast.unparse(kw.value),f.name)
                                count+=1;covered.add(f.name)
        self.assertEqual(covered,names);self.assertEqual(count,20)

    def test_standard_webhook_paths_wrap_but_do_not_change_saved_source(self):
        code=Path('src/app.jsx').read_text()
        self.assertIn('notes: prepareShareableNotesInput(text)',code)
        self.assertIn('notes: prepareShareableNotesInput(transcriptText)',code)
        self.assertNotIn('rawNotes: prepareShareableNotesInput',code)
