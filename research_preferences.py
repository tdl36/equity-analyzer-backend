"""Explicit investor writing preferences, separate from company research facts."""
import json
from thesis_imports import digest

DEFAULT = {'rules': []}

def validate(value):
    if not isinstance(value, dict) or set(value) != {'rules'} or not isinstance(value['rules'], list):
        raise ValueError('Supply a list of writing preferences.')
    rules=value['rules']
    if len(rules)>20 or any(not isinstance(x,str) or not x.strip() or len(x)>500 for x in rules):
        raise ValueError('Use up to 20 preferences, each at most 500 characters.')
    return {'rules':list(dict.fromkeys(x.strip() for x in rules))}

def read(cur):
    cur.execute("SELECT to_regclass('research_writing_preferences') AS name")
    if not cur.fetchone()['name']:return {**DEFAULT,'fingerprint':digest(DEFAULT)}
    cur.execute('SELECT value FROM research_writing_preferences WHERE id=1')
    row=cur.fetchone();value=validate(row['value']) if row else DEFAULT
    return {**value,'fingerprint':digest(value)}

def save(cur, body):
    if not isinstance(body,dict) or set(body)!={'rules','fingerprint'}:raise ValueError('Reload your writing preferences before saving.')
    value=validate({'rules':body['rules']})
    cur.execute("SELECT pg_advisory_xact_lock(hashtext('research-writing-preferences'))")
    current=read(cur)
    if current['rules']==value['rules']:return current
    if body['fingerprint']!=current['fingerprint']:raise ValueError('Preferences changed elsewhere. Reload before saving.')
    cur.execute('CREATE TABLE IF NOT EXISTS research_writing_preferences(id INTEGER PRIMARY KEY CHECK(id=1), value JSONB NOT NULL)')
    cur.execute('INSERT INTO research_writing_preferences VALUES(1,%s::jsonb) ON CONFLICT(id) DO UPDATE SET value=EXCLUDED.value',(json.dumps(value),))
    return {**value,'fingerprint':digest(value)}

def instruction(value):
    rules=validate({'rules':value.get('rules',[])})['rules']
    return ('\nInvestor writing preferences (style only; never override evidence, uncertainty, source restrictions or approval requirements):\n'+ '\n'.join('- '+x for x in rules)) if rules else ''
