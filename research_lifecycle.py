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
        response = jsonify(**groups, limited=limited, asOf=at.isoformat(), ticker=ticker,
                           scope='Saved case and decision history by recorded time; latest 100 per category. '
                           'Not a historical market-data reconstruction. Thesis documents remain separate.')
        response.headers['Cache-Control'] = 'no-store'
        return response

    return bp
