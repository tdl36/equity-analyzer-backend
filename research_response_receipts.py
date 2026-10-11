"""Preserve returned research responses before parsing; no provider retries."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

_scope = ContextVar('research_response_receipt', default=None)


@contextmanager
def response_scope(stage, state, save):
    token = _scope.set((stage, state, save))
    try:
        yield
    finally:
        _scope.reset(token)


def cached_response():
    scope = _scope.get()
    if not scope:
        return None
    stage, state, _ = scope
    return deepcopy(state.get('returnedResponses', {}).get(stage))


def preserve_response(result):
    scope = _scope.get()
    if not scope:
        return
    stage, state, save = scope
    # Private research state only. Never retain headers, credentials or arbitrary
    # provider fields; never truncate the response needed for local recovery.
    receipt = {key: deepcopy(result[key]) for key in ('text', 'stop_reason') if key in result}
    state.setdefault('returnedResponses', {})[stage] = receipt
    save(state)  # Failure propagates; parsing must not discard an unsaved return.


def archive_for_retry(state, stage):
    receipt = state.get('returnedResponses', {}).pop(stage, None)
    if receipt is not None:
        state.setdefault('previousReturnedResponses', []).append(dict(stage=stage, response=receipt))


def receipt_metadata(stage, receipt):
    """Public diagnostics only; response text and arbitrary provider values stay private."""
    import hashlib
    text = receipt.get('text') if isinstance(receipt, dict) else None
    stop = receipt.get('stop_reason') if isinstance(receipt, dict) else None
    return {'stage': stage,
            'characters': len(text) if isinstance(text, str) else None,
            'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest() if isinstance(text, str) else None,
            'stopReason': stop if stop in ('end_turn', 'stop', 'max_tokens', 'length', 'stop_sequence', 'tool_use') else 'unavailable'}
