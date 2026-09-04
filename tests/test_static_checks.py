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


def test_server_note_prompt_keeps_the_profit_vs_revenue_guidance():
    """The two note prompts must not drift apart on the profit chart.

    The agent's prompt spelled out that profit means operating income and that
    an empty array beats reusing revenue. When generation moved server-side that
    paragraph did not come with it, leaving the server asking only for "a JSON
    array of profit segments" -- which a model answers most easily by repeating
    the revenue split. segment_charts.is_duplicate_series then suppresses the
    chart, so the visible symptom is a missing profit donut, several steps away
    from the prompt that caused it.
    """
    server = (ROOT / 'app_v3.py').read_text(encoding='utf-8')
    agent = (ROOT / 'charlie_local_agent.py').read_text(encoding='utf-8')
    required = [
        'Revenue and profit MUST be different numbers',
        'NOT revenue',
        'return an empty array',
    ]
    for phrase in required:
        assert phrase in agent, f'agent prompt lost its guidance: {phrase!r}'
        assert phrase in server, (
            f'app_v3.py note prompt is missing {phrase!r} -- the server is the '
            'default note path, so it needs the same guidance as the agent')


def test_note_and_thesis_models_are_not_hardcoded():
    """Both must resolve the model the user picked, not a baked-in id.

    The thesis analysis was pinned to claude-sonnet-4-5-20250929 and the
    reconciler to claude-sonnet-4-6, so the picker could offer Opus or Fable and
    the run would quietly use neither. A literal here is also how a retirement
    becomes an outage: the same shape took out note generation when
    claude-sonnet-4-20250514 was withdrawn.
    """
    src = (ROOT / 'app_v3.py').read_text(encoding='utf-8').splitlines()
    offenders = []
    for fn_start, fn_name in _function_spans(src, (
            '_run_analysis_job', '_reconcile_stale_facts', '_generate_research_note',
            '_run_decipher_job', '_run_decipher_followup_job',
            'mp_run_pipeline', 'mp_job_retry', 'mp_save_results')):
        for lineno, line in fn_start:
            if re.search(r"model\s*=\s*['\"]claude-", line):
                offenders.append(f'{fn_name} (app_v3.py:{lineno}): {line.strip()}')
    assert not offenders, (
        'model id hardcoded where the user picks one:\n  ' + '\n  '.join(offenders))


def _function_spans(lines, names):
    """Yield (numbered lines, name) for each named top-level function."""
    starts = {}
    for i, line in enumerate(lines):
        m = re.match(r'def (\w+)\(', line)
        if m and m.group(1) in names:
            starts[i] = m.group(1)
    out = []
    for start, name in starts.items():
        body = []
        for i in range(start + 1, len(lines)):
            if re.match(r'def \w+\(', lines[i]):
                break
            body.append((i + 1, lines[i]))
        out.append((body, name))
    return out


def test_models_endpoint_lists_the_picker_models():
    """The picker is served, not baked into the bundle."""
    import app_v3
    assert app_v3.PICKER_MODELS, 'no models offered'
    keys = {m['key'] for m in app_v3.PICKER_MODELS}
    assert app_v3.PICKER_DEFAULT_MODEL in keys, 'default is not one of the options'
    for m in app_v3.PICKER_MODELS:
        assert m.get('model', '').startswith('claude-'), m
        assert m.get('label'), m
    # a stale key must degrade, never fail the job
    assert app_v3.resolve_picker_model('no-such-model') == \
        app_v3.resolve_picker_model(app_v3.PICKER_DEFAULT_MODEL)


def test_every_feature_default_is_a_real_picker_option():
    """A per-feature default that is not in the list would be unselectable.

    Decipher runs on Opus 4.7 and meeting prep on Sonnet 5. If either default
    named a key the picker does not offer, resolve_picker_model would silently
    substitute the global default and the feature would change model without
    anyone choosing that.
    """
    import app_v3
    keys = {m['key'] for m in app_v3.PICKER_MODELS}
    for name in ('PICKER_DEFAULT_MODEL', 'DECIPHER_DEFAULT_MODEL',
                 'MEETING_PREP_DEFAULT_MODEL'):
        key = getattr(app_v3, name)
        assert key in keys, f'{name}={key!r} is not one of {sorted(keys)}'


def test_feature_defaults_preserve_the_model_each_feature_already_used():
    """Adding a picker must not move a feature to a different model.

    Decipher was pinned to claude-opus-4-7 in four places. If its default
    resolved to anything else, every Explain answer would quietly change on
    deploy -- a regression that looks like a feature.
    """
    import app_v3
    assert app_v3.resolve_picker_model(None, app_v3.DECIPHER_DEFAULT_MODEL) \
        == 'claude-opus-4-7'


def test_no_dated_model_ids_outside_the_registry():
    """Dated ids are the ones that get retired.

    claude-sonnet-4-20250514 was withdrawn and took out three features at once,
    each found by someone hitting it. The registry may name a dated id; nothing
    else should, so a retirement is one edit in one place.
    """
    src = (ROOT / 'app_v3.py').read_text(encoding='utf-8').splitlines()
    registry = False
    offenders = []
    for i, line in enumerate(src, 1):
        if 'PICKER_MODELS = [' in line or 'MODEL_TIERS = {' in line:
            registry = True
        elif registry and line.startswith((']', '}')):
            registry = False
        elif registry:
            continue
        if 'os.environ.get' in line or line.lstrip().startswith('#'):
            continue
        # Two tables legitimately name ids as data rather than calling them: the
        # per-model cost lookup, and the fallback list used when the provider's
        # list-models endpoint is unreachable. A retirement makes an entry
        # useless, not fatal.
        if '_PRICE' in line or ': 0.0' in line or line.strip().startswith('"claude-'):
            continue
        if re.search(r"['\"]claude-[a-z0-9.-]*-\d{8}['\"]", line):
            offenders.append(f'app_v3.py:{i}: {line.strip()[:90]}')
    assert not offenders, (
        'dated model id outside the registry:\n  ' + '\n  '.join(offenders))


def test_the_agent_restarts_itself_when_its_source_changes():
    """launchd keeps the agent alive forever, so an edit never lands by itself.

    This one ran thirteen days past a model-id fix, failing every note job with
    a 404 for a model that had been retired, while the corrected line sat on
    disk unread. Nothing anywhere reported the mismatch.
    """
    src = (ROOT / 'charlie_local_agent.py').read_text(encoding='utf-8')
    assert '_source_changed' in src, 'no source-change check'
    assert 'getmtime' in src, 'the check does not look at the file'
    # and it must actually be called from the poll loop, not merely defined
    body = src[src.index('def _source_changed'):]
    assert body.count('_source_changed()') >= 1, (
        '_source_changed is defined but never called')
