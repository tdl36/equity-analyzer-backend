"""Count how many signpost rows actually rendered on memo page 2.

Row loss was silent: the page count stayed 3 and the preflight passed while two
rows fell off the bottom. Extraction drops hyphens at line breaks, so names are
compared on alphanumerics only.
"""
import json, re, subprocess, sys

pdf, fixture = sys.argv[1], sys.argv[2]
path = {'de': 'fixtures/deepdive_de_golden.json',
        'unh': 'fixtures/deepdive_unh_sample.json',
        'stress': 'fixtures/deepdive_stress.json'}[fixture]
names = [sp.get('signpost', '') for sp in
         (json.load(open(path))['master'].get('signposts') or [])]
txt = subprocess.run(['pdftotext', '-f', '2', '-l', '2', pdf, '-'],
                     capture_output=True, text=True).stdout
flat = re.sub(r'[^a-z0-9]', '', txt.lower())
print(sum(1 for n in names if n and re.sub(r'[^a-z0-9]', '', n.lower()) in flat))
