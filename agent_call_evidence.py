"""Content-free, run-scoped model callback evidence. Never authorizes replay."""
import json
import re
from datetime import datetime, timezone
from uuid import UUID


def append(get_db, run_id, execution_id, call_id, outcome, response=None):
    entry = dict(event='model_call_v1', agent='system',
                 executionId=str(UUID(execution_id)), callId=str(UUID(str(call_id))),
                 outcome=outcome, ts=datetime.now(timezone.utc).isoformat(),
                 message='Graph model call evidence: ' + outcome)
    if outcome not in ('started', 'returned', 'error'):
        raise ValueError('Unsupported model call outcome')
    # Model text, prompts, headers, exception bodies and arbitrary metadata never
    # enter the evidence ledger. These IDs identify responses, not HTTP requests.
    ids = []
    for group in getattr(response, 'generations', []) or []:
        for generation in group:
            identifier = getattr(getattr(generation, 'message', None), 'id', None)
            if isinstance(identifier, str) and re.fullmatch(r'(?:chatcmpl-|msg_|resp_)[A-Za-z0-9_-]{1,160}', identifier):
                if identifier not in ids:
                    ids.append(identifier)
    entry['responseIds'] = ids[:5]
    with get_db(commit=True) as (_, cur):
        cur.execute("UPDATE agent_runs SET logs=COALESCE(logs,'[]'::jsonb) || %s::jsonb WHERE id=%s RETURNING id",
                    (json.dumps([entry]), run_id))
        if not cur.fetchone():
            raise ValueError('Model call evidence could not be saved')


def attach(graph, get_db, run_id, execution_id):
    """Attach to this instance's graph args, including rebuilt checkpoint graphs.

    Coverage is graph model callbacks only; initialization and final signal
    processing may call providers separately. Never infer whole-run completion.
    """
    from langchain_core.callbacks.base import BaseCallbackHandler

    class Evidence(BaseCallbackHandler):
        raise_error = True
        run_inline = True

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            self.on_llm_start(serialized, None, run_id=run_id)

        def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
            append(get_db, saved_run, execution_id, run_id, 'started')

        def on_llm_end(self, response, *, run_id, **kwargs):
            append(get_db, saved_run, execution_id, run_id, 'returned', response)

        def on_llm_error(self, error, *, run_id, **kwargs):
            append(get_db, saved_run, execution_id, run_id, 'error')

    saved_run = run_id
    original = graph.propagator.get_graph_args
    handler = Evidence()

    def args_with_evidence(*args, **kwargs):
        args = dict(original(*args, **kwargs))
        config = dict(args.get('config') or {})
        callbacks = config.get('callbacks') or []
        if not isinstance(callbacks, (list, tuple)):
            raise ValueError('Unsupported graph callback configuration')
        config['callbacks'] = [*callbacks, handler]
        args['config'] = config
        return args

    graph.propagator.get_graph_args = args_with_evidence


def summary(entries):
    calls = {}
    for entry in entries:
        if entry.get('event') != 'model_call_v1':
            continue
        try:
            key = (str(UUID(str(entry.get('executionId')))), str(UUID(str(entry.get('callId')))))
        except (ValueError, TypeError, AttributeError):
            continue
        outcome = entry.get('outcome')
        if outcome in ('started', 'returned', 'error'):
            calls.setdefault(key, set()).add(outcome)
    return dict(coverage='graph_callbacks_only', recordedCalls=len(calls),
                returned=sum('returned' in states for states in calls.values()),
                errors=sum('error' in states for states in calls.values()),
                unresolved=sum(states == {'started'} for states in calls.values()))
