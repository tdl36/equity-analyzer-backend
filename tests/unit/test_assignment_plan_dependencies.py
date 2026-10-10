"""Mac command validation must work without server/site-package dependencies."""
import subprocess
import sys
import unittest
from pathlib import Path

class AssignmentDependenciesTests(unittest.TestCase):
    def test_command_planner_without_site_packages(self):
        root=Path(__file__).resolve().parents[2]
        code="""
from research_assignment_plan import plan
p=plan({'ticker':'UNH','since':'2026-08-01','until':'2026-10-09','outputs':['thesis','visual']})
assert p['ticker']=='UNH' and p['since']=='2026-08-01'
"""
        result=subprocess.run([sys.executable,'-S','-c',code],cwd=root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        collector=(root/'collection_refresh.py').read_text()
        self.assertNotIn('from research_assignments import plan',collector)
