"""Static checks that run with the suite.

Added after shipping `notegen.plan_batches(...)` to production with no
`import notegen`: an edit meant for app_v3.py matched a string that only exists
in deepdive.py, so the import went nowhere. 363 tests passed, because none of
them touched that code path, and the failure surfaced as a NameError in a
background thread on a live run.

pyflakes finds that in under a second. Every test below is cheap and catches a
class of mistake that unit tests miss precisely because they never execute the
line.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Modules that run in production. Deliberately explicit rather than a glob:
# a new backend module should be added here consciously.
PRODUCTION_MODULES = [
    'app_v3.py', 'notegen.py', 'segment_charts.py', 'deepdive.py',
    'deepdive_prompts.py', 'onepager.py', 'signposts.py', 'thesis_history.py',
    'briefings.py', 'earnings.py', 'sec_edgar.py', 'theme_tracker.py',
    'charlie_local_agent.py',
]


def _pyflakes(paths):
    proc = subprocess.run(
        [sys.executable, '-m', 'pyflakes', *paths],
        capture_output=True, text=True, cwd=ROOT, timeout=180)
    return proc.stdout.splitlines()


def test_no_undefined_names_anywhere_in_production_code():
    """A NameError in a background thread is found by a user, not a test."""
    present = [m for m in PRODUCTION_MODULES if (ROOT / m).exists()]
    undefined = [ln for ln in _pyflakes(present) if 'undefined name' in ln]
    assert not undefined, (
        'undefined name(s) — these raise NameError when the line runs:\n  '
        + '\n  '.join(undefined))


def test_no_undefined_names_in_the_test_suite_itself():
    """The tests missed a missing import; they can have one too."""
    tests = [str(p.relative_to(ROOT)) for p in (ROOT / 'tests').glob('test_*.py')]
    undefined = [ln for ln in _pyflakes(tests) if 'undefined name' in ln]
    assert not undefined, 'undefined name(s) in tests:\n  ' + '\n  '.join(undefined)


# --- dependencies -----------------------------------------------------------
#
# matplotlib, openpyxl and yfinance were all imported at runtime and never
# declared. Each import sits inside a try/except, so the failure mode is not a
# crash: charts silently absent, a selected .xlsx silently ignored, prices
# silently empty. Nothing in the app says the feature is off.

def _third_party_imports(paths):
    import ast
    import sys as _sys
    stdlib = set(_sys.stdlib_module_names)
    local = {p.stem for p in ROOT.glob('*.py')} | {
        p.name for p in ROOT.iterdir() if p.is_dir() and (p / '__init__.py').exists()}
    found = {}
    for rel in paths:
        p = ROOT / rel
        if not p.exists():
            continue
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                names = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split('.')[0]]
            else:
                continue
            for n in names:
                if n not in stdlib and n not in local:
                    found.setdefault(n, set()).add(rel)
    return found


# Imports that are genuinely optional at runtime and may be absent by design.
OPTIONAL_IMPORTS = {
    'tradingagents',   # heavy optional integration, guarded at every call site
    # Supplied transitively by tradingagents, which requires yfinance>=1.4.1.
    # Pinning it here made requirements.txt unresolvable and the Render build
    # failed -- silently, because a failed build leaves the previous process
    # serving, so the deploy simply never lands and nothing says so.
    'yfinance',
}


# An import name is not always the distribution name.
IMPORT_TO_DISTRIBUTION = {
    'psycopg2': 'psycopg2-binary', 'google': 'google-genai', 'PIL': 'pillow',
    'yaml': 'pyyaml', 'dotenv': 'python-dotenv', 'dateutil': 'python-dateutil',
    'fitz': 'pymupdf', 'docx': 'python-docx', 'pptx': 'python-pptx',
    'bs4': 'beautifulsoup4', 'jwt': 'pyjwt', 'sklearn': 'scikit-learn',
    'tavily': 'tavily-python', 'flask_cors': 'flask-cors',
    'youtube_transcript_api': 'youtube-transcript-api', 'apscheduler': 'apscheduler',
    'googleapiclient': 'google-api-python-client', 'google_auth': 'google-auth',
    'pywebpush': 'pywebpush',
}


def _declared_distributions():
    text = (ROOT / 'requirements.txt').read_text(encoding='utf-8').lower()
    names = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # strip version pins, extras, and git/url forms
        name = re.split(r'[<>=!\[@; ]', line, 1)[0].strip()
        if name:
            names.add(name)
            names.add(name.replace('-', '_'))
    return names


def test_every_import_in_production_code_is_declared():
    """An undeclared dependency degrades a feature silently, not loudly.

    Checked against requirements.txt rather than the local virtualenv: Render
    installs from the file, and a developer machine can happen to have a package
    that production will not. matplotlib was exactly that -- present locally on
    the agent's machine, absent on the server, and the only symptom was a note
    that came out with no charts.
    """
    declared = _declared_distributions()
    undeclared = []
    for module, used_in in sorted(_third_party_imports(PRODUCTION_MODULES).items()):
        if module in OPTIONAL_IMPORTS:
            continue
        dist = IMPORT_TO_DISTRIBUTION.get(module, module).lower()
        if dist not in declared and module.lower() not in declared:
            undeclared.append(f'{module} (used in {", ".join(sorted(used_in))})')
    assert not undeclared, (
        'import(s) missing from requirements.txt — these fail silently in '
        'production because each is wrapped in a try/except:\n  '
        + '\n  '.join(undeclared))


# --- retired models ---------------------------------------------------------
#
# claude-sonnet-4-20250514 was referenced in three places when it was retired.
# The API answers 404, _is_retryable treated that as fatal, and three features
# broke independently -- each found by a person using them, not by a test.

RETIRED_MODEL_IDS = {
    'claude-sonnet-4-20250514',
}

MODEL_BEARING_FILES = [
    'app_v3.py', 'charlie_local_agent.py', 'deepdive.py', 'onepager.py',
    'briefings.py', 'earnings.py',
]


def test_no_retired_model_ids_in_live_code():
    """A retired id is a 404 waiting for whoever uses that feature next."""
    offenders = []
    for rel in MODEL_BEARING_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith('#'):        # explanatory comments are fine
                continue
            for dead in RETIRED_MODEL_IDS:
                if dead in line:
                    offenders.append(f'{rel}:{i}: {stripped[:80]}')
    assert not offenders, 'retired model id(s) still referenced:\n  ' + '\n  '.join(offenders)


def test_a_404_falls_through_to_the_next_provider():
    """A model this provider cannot serve must not fail the whole call.

    _is_retryable treated any non-(429,5xx) Anthropic status as fatal, so one
    stale model id broke the chain instead of letting Gemini answer.
    """
    import anthropic as _anthropic
    import httpx
    import app_v3

    def _err(cls, status):
        request = httpx.Request('POST', 'https://api.anthropic.com/v1/messages')
        response = httpx.Response(status, request=request)
        return cls('boom', response=response, body=None)

    assert app_v3._is_retryable('anthropic', _err(_anthropic.NotFoundError, 404)) is True

    assert app_v3._is_retryable(
        'anthropic', _err(_anthropic.AuthenticationError, 401)) is False, (
        'an auth failure fails on every provider; it must stay fatal')


def test_no_invalid_escape_sequences():
    """These are a SyntaxWarning today and a SyntaxError in a later Python.

    Two were hiding in a dead copy of the donut renderer -- code that no longer
    ran, so nothing surfaced them, and that would have broken the build on a
    Python upgrade for a function nobody used.
    """
    import warnings
    offenders = []
    for rel in PRODUCTION_MODULES:
        path = ROOT / rel
        if not path.exists():
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', SyntaxWarning)
            compile(path.read_text(encoding='utf-8'), rel, 'exec')
        for w in caught:
            if issubclass(w.category, SyntaxWarning):
                offenders.append(f'{rel}: {w.message}')
    assert not offenders, 'invalid escape sequence(s):\n  ' + '\n  '.join(offenders)


def test_requirements_resolve_together():
    """The file must install as a set, not one package at a time.

    Adding a yfinance pin looked fine because `pip install yfinance` succeeded
    into an already-populated virtualenv. As a set it was unresolvable --
    tradingagents requires yfinance>=1.4.1 -- and the Render build failed. A
    failed build keeps the old process running, so the symptom was a deploy that
    never arrived rather than an error.
    """
    proc = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--dry-run', '-q',
         '-r', str(ROOT / 'requirements.txt')],
        capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert proc.returncode == 0, (
        'requirements.txt does not resolve:\n'
        + (proc.stderr or proc.stdout)[-1500:])
