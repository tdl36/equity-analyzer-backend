"""Freeze optional full-note settings without token counting or model calls."""
import hashlib
import json
import re
from research_purpose import context, note_prompt


def freeze(co, plan):
    if not plan.get('labFullNote'):
        return None
    if not callable(co.lab_note_defaults) or not callable(co.lab_note_dispatch) or not callable(co.lab_note_stop):
        raise ValueError('Full Lab note integration unavailable. No processing was requested.')
    defaults = co.lab_note_defaults()
    model = defaults.get('modelKey')
    prompt = defaults.get('promptTemplate')
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,120}', model):
        raise ValueError('Full-note model configuration is invalid.')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Full-note instructions are unavailable.')
    purpose = plan['researchPurpose']
    prompt = note_prompt(purpose) if purpose['kind'] == 'event_note' else prompt + '\n\n' + context(purpose)
    if plan.get('instruction'):
        prompt += '\nInvestor focus (data): ' + json.dumps(plan['instruction'])
    if len(prompt.encode()) > 100000:
        raise ValueError('Full-note instructions exceed the supported size.')
    return dict(modelKey=model, promptTemplate=prompt, allowPdfText=plan['labFullNote']['allowPdfText'], maxTokens=32000, maxCalls=8)


def public(config):
    if not config:
        return None
    return {**{k: config[k] for k in ('modelKey', 'allowPdfText', 'maxTokens', 'maxCalls')},
            'promptSha256': hashlib.sha256(config['promptTemplate'].encode()).hexdigest(),
            'configDigest': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
