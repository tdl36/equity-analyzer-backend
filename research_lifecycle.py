"""Read-only reconstruction of saved investment decisions at a recorded-time cutoff."""
import re
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request


def cutoff(value):
    if not value:
        return datetime.now(timezone.utc)
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        raise ValueError('Choose a valid recorded-time cutoff.')
    if result.tzinfo is None or result > datetime.now(timezone.utc):
        raise ValueError('Use a past or current timestamp with a timezone.')
    return result


def create_blueprint(get_db):
    bp = Blueprint('research_lifecycle', __name__)

    @bp.get('/api/research/lifecycle/<ticker>')
    def lifecycle(ticker):
        if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,19}', ticker):
            return jsonify(error='Choose a valid ticker.'), 400
        try:
            at = cutoff(request.args.get('asOf'))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        groups, limited = {}, {}
        with get_db() as (_, cur):
            # Fixed identifiers only. Never interpret a document date as knowledge time.
            for key, table in [('cases', 'investment_case_versions'),
                               ('work', 'research_work_versions'),
                               ('decisions', 'research_decisions')]:
                cur.execute('SELECT to_regclass(%s) AS name', (table,))
                if not cur.fetchone()['name']:
                    groups[key], limited[key] = [], False
                    continue
                cur.execute(f'SELECT * FROM {table} WHERE ticker=%s AND created_at<=%s '
                            'ORDER BY created_at DESC,revision DESC LIMIT 101', (ticker, at))
                rows = [dict(row) for row in cur.fetchall()]
                limited[key] = len(rows) > 100
                groups[key] = [{k: r[k] for k in ('id', 'revision', 'body', 'created_at') if k in r}
                               for r in rows[:100]]
            cur.execute("SELECT to_regclass('mp_jobs') AS name")
            groups['proposalReviews']=[]
            if cur.fetchone()['name']:
                cur.execute("""SELECT id,input->'baseline'->'_investmentCase'->>'revision' AS case_revision,
                    result->'reviewDecision' AS review,created_at FROM mp_jobs
                    WHERE ticker=%s AND stage='evidence_amendment' AND input->>'target'='investment_case'
                    AND result->'reviewDecision'->>'recordedAt' IS NOT NULL
                    AND (result->'reviewDecision'->>'recordedAt')::timestamptz<=%s
                    ORDER BY result->'reviewDecision'->>'recordedAt' DESC LIMIT 101""",(ticker,at))
                reviews=[dict(r) for r in cur.fetchall()]
                limited['proposalReviews']=len(reviews)>100
                groups['proposalReviews']=reviews[:100]
        response = jsonify(**groups, limited=limited, asOf=at.isoformat(), ticker=ticker,
                           scope='Saved case and decision history by recorded time; latest 100 per category. '
                           'Not a historical market-data reconstruction. Thesis documents remain separate.')
        response.headers['Cache-Control'] = 'no-store'
        return response

    @bp.get('/api/research/underweights')
    def underweights():
        with get_db() as (_,cur):
            cur.execute("SELECT to_regclass('research_work_versions') AS name")
            if not cur.fetchone()['name']:return jsonify(records=[],limited=False)
            cur.execute("""SELECT * FROM
                (SELECT DISTINCT ON(id) id,ticker,revision,body,created_at FROM research_work_versions ORDER BY id,revision DESC) latest
                WHERE body->>'kind'='underweight' ORDER BY body->>'dueDate',ticker,id LIMIT 201""")
            rows=[dict(r) for r in cur.fetchall()]
        response=jsonify(records=rows[:200],limited=len(rows)>200)
        response.headers['Cache-Control']='no-store'
        return response

    return bp
