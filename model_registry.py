"""Reviewed, release-pinned model policy shared by server and Mac agent.

Discovery is not approval: provider catalogs contain neither trustworthy prices
nor migration compatibility. The scheduled maintainer verifies official sources,
validates this policy and ships tested releases. No runtime lexical ID guessing.
"""
import copy
import hashlib
import json
import os
import math
from urllib.parse import urlparse
from datetime import date, datetime, timezone
from pathlib import Path

PATH = Path(__file__).with_suffix('.json')
REGISTRY = json.loads(PATH.read_text())
REVISION = hashlib.sha256(PATH.read_bytes()).hexdigest()[:16]
MODELS = REGISTRY['models']
PRICES = {m: (s['input'], s['output']) for m, s in MODELS.items() if 'input' in s and 'output' in s}


def role(name, env=None):
    return (os.environ.get(env) if env else None) or REGISTRY['roles'][name]


def ensure_active(model, today=None):
    retirement = MODELS.get(model, {}).get('retirement')
    if retirement and (today or datetime.now(timezone.utc).date()).isoformat() >= retirement:
        raise ValueError(f'{model} retired on {retirement}. Choose an active model; automatic substitution is disabled for pinned jobs.')
    return model


def request_options(model):
    ensure_active(model)
    return copy.deepcopy(MODELS.get(model, {}).get('request', {}))


def picker_models(today=None):
    today = (today or datetime.now(timezone.utc).date()).isoformat()
    return copy.deepcopy([p for p in REGISTRY['picker'] if MODELS[p['model']].get('retirement', '9999-12-31') > today])


def public_status(today=None):
    today = today or datetime.now(timezone.utc).date()
    notices = []
    for model, spec in MODELS.items():
        if spec.get('retirement'):
            left = (date.fromisoformat(spec['retirement']) - today).days
            notices.append({'model': model, 'date': spec['retirement'], 'days': left,
                            'message': 'Retired; selection blocked' if left <= 0 else 'Replacement needs review before retirement'})
    return {'revision': REVISION, 'verifiedAt': REGISTRY['verifiedAt'],
            'roles': REGISTRY['roles'], 'models': MODELS, 'notices': notices,
            'history': REGISTRY.get('history', [])[-20:], 'sources': REGISTRY['sources'],
            'policy': 'Routine replacements require verified prices, compatible requests and no higher modeled cost. Research defaults stay pinned. Per-job bills vary with token use.',
            'maintenanceDependency': 'Scheduled maintenance requires this Mac, Codex, network access and deployment credentials. Runtime uses the last deployed registry.'}


def usage_dict(usage, provider='anthropic'):
    def get(obj, key, default=0):
        return (obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)) or default
    if provider == 'openai':
        return {'input_tokens': get(usage, 'prompt_tokens'), 'output_tokens': get(usage, 'completion_tokens'),
                'cache_read_input_tokens': get(get(usage, 'prompt_tokens_details', {}), 'cached_tokens')}
    creation = get(usage, 'cache_creation', {})
    return {'input_tokens': get(usage, 'input_tokens'), 'output_tokens': get(usage, 'output_tokens'),
            'cache_read_input_tokens': get(usage, 'cache_read_input_tokens'),
            'cache_creation_input_tokens': get(usage, 'cache_creation_input_tokens'),
            'cache_creation_1h_input_tokens': get(creation, 'ephemeral_1h_input_tokens')}


def estimate(model, usage):
    """USD estimate or None, never zero for an unknown price. Thinking is output.

    Anthropic input excludes cache; OpenAI input includes cache reads. Tool fees,
    special service tiers and image generation are outside this text estimate.
    """
    spec = MODELS.get(model, {})
    if 'input' not in spec or 'output' not in spec:
        return None
    tin, tout = max(0, usage.get('input_tokens', 0) or 0), max(0, usage.get('output_tokens', 0) or 0)
    read = max(0, usage.get('cache_read_input_tokens', 0) or 0)
    write = max(0, usage.get('cache_creation_input_tokens', 0) or 0)
    hour = min(write, max(0, usage.get('cache_creation_1h_input_tokens', 0) or 0))
    if spec['provider'] == 'openai':
        read = min(tin, read)
        tin -= read
    context = tin + read + write
    long = spec.get('longContext', {})
    im = long.get('inputMultiplier', 1) if context > long.get('threshold', float('inf')) else 1
    om = long.get('outputMultiplier', 1) if context > long.get('threshold', float('inf')) else 1
    cost = im * (tin * spec['input'] + read * spec.get('cacheRead', spec['input']) +
                 (write-hour) * spec.get('cacheWrite5m', spec['input']) + hour * spec.get('cacheWrite1h', spec['input']))
    return round((cost + tout * spec['output'] * om) / 1e6, 6)


def validate(data):
    errors = []
    for name, model in data['roles'].items():
        if model not in data['models']: errors.append(f'{name}: unknown model {model}')
    for entry in data['picker']:
        spec = data['models'].get(entry['model'], {})
        if spec.get('provider') != entry['provider'] or not all(isinstance(spec.get(k), (float, int)) and spec[k] > 0 for k in ('input', 'output')):
            errors.append(f"{entry['key']}: missing provider or prices")
    if len({p['key'] for p in data['picker']}) != len(data['picker']): errors.append('duplicate picker key')
    for model, spec in data['models'].items():
        if spec.get('retirement'): date.fromisoformat(spec['retirement'])
        opts = spec.get('request', {})
        if model == 'claude-sonnet-5-5' and opts.get('thinking', {}).get('type') != 'between_tools':
            errors.append('Sonnet 5.5 routine policy requires between_tools')
    return errors


def migration_errors(before, after, evidence):
    """Fail closed for unattended default changes. Evidence is human/agent-reviewed
    official documentation, not fabricated by a provider catalog listing.
    """
    errors = validate(after)
    if set(before['roles']) != set(after['roles']):
        errors.append('Adding or removing workflow roles requires review')
    if set(before['protectedRoles']) != set(after['protectedRoles']):
        errors.append('Changing protected roles requires investor approval')
    for name, new in after['roles'].items():
        old = before['roles'].get(name)
        if old == new and before['models'].get(old, {}).get('request', {}) == after['models'].get(new, {}).get('request', {}): continue
        if name in before['protectedRoles']:
            errors.append(f'{name}: protected default requires investor approval'); continue
        proof = evidence.get(name, {})
        a, b = before['models'].get(old, {}), after['models'].get(new, {})
        if not all(proof.get(k) for k in ('officialSources', 'accountAvailable', 'compatibilityTested', 'qualityEvidence', 'tokenMultiplier')):
            errors.append(f'{name}: incomplete evidence'); continue
        if not isinstance(proof['officialSources'], list) or any(urlparse(url).scheme != 'https' or urlparse(url).hostname not in ('platform.claude.com', 'www.anthropic.com', 'anthropic.com', 'developers.openai.com', 'platform.openai.com', 'openai.com') for url in proof['officialSources']):
            errors.append(f'{name}: evidence must link official sources')
        multiplier = proof['tokenMultiplier']
        if isinstance(multiplier, bool) or not isinstance(multiplier, (float, int)) or not math.isfinite(multiplier) or multiplier < 1:
            errors.append(f'{name}: invalid token multiplier'); continue
        if a.get('provider') != b.get('provider') or not set(a.get('capabilities', [])).issubset(b.get('capabilities', [])):
            errors.append(f'{name}: provider or source capability regression')
        for price in ('input', 'output', 'cacheRead', 'cacheWrite5m', 'cacheWrite1h'):
            if price in a and (price not in b or b[price] * multiplier > a[price]):
                errors.append(f'{name}: higher or unknown {price} cost')
        if not all(k in b for k in ('input','output')): errors.append(f'{name}: unknown price')
        if b.get('longContext', {}) != a.get('longContext', {}): errors.append(f'{name}: changed long-context pricing needs review')
        if proof.get('noAdditionalThinking') is not True: errors.append(f'{name}: thinking cost uncertain')
        if proof.get('sameOrLowerTokenCaps') is not True: errors.append(f'{name}: token caps increased')
    return errors
