"""Authenticated pending historical diagnostics, never approval or paid dispatch.

Production exposes read-only inspection; ingestion and proposal writes stay disabled.
"""
import json
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request
from historical_valuation import _source, _date
from historical_valuation_context import load_context
from historical_valuation_source_store import read_originals
from historical_valuation_store import _identity, save_pending, read_pending
from historical_valuation_retrieval import retrieve_once, inspect_retrievals


def references(proposal):
    if not isinstance(proposal, dict):
        raise ValueError('A historical proposal is required.')
    rows = proposal.get('observations')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 30:
        raise ValueError('One to thirty historical observations are required.')
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid historical observation.')
        for field in ('price', 'earnings'):
            value = row.get(field)
            receipt = value.get('source') if isinstance(value, dict) else None
            if not _source(receipt):
                raise ValueError('Historical observations require exact source receipts.')
            result.append({k:receipt[k] for k in ('id', 'sha256')})
    return result


def create_blueprint(get_db, authenticated=lambda: False, *, ingestion_config=None, read_only=False):
    bp = Blueprint('historical_valuation_review', __name__)

    @bp.before_request
    def readonly_gate():
        if read_only and request.method not in ('GET', 'HEAD', 'OPTIONS'):
            if not authenticated():
                return jsonify(error='Sign in to inspect historical evidence.'), 403
            return jsonify(error='Historical evidence is read-only. Retrieval and proposal creation are not enabled.'), 405

    @bp.after_request
    def private(response):
        response.headers['Cache-Control'] = 'private, no-store'
        return response

    def bounded_selection(fields):
        if set(request.args) != {'selection'} or len(request.args.getlist('selection')) != 1:
            raise ValueError('One exact selection is required.')
        raw = request.args['selection']
        if len(raw.encode()) > 4096:
            raise ValueError('Selection exceeds the bound.')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Duplicate selection key.')
                result[key] = value
            return result
        def nonfinite(value):
            raise ValueError('Nonfinite selection.')
        selection = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
        if not isinstance(selection, dict) or set(selection) != fields:
            raise ValueError('Select a frozen baseline and exact inputs.')
        return selection

    @bp.get('/api/research/assignments/<aid>/onepager-editions/<eid>/historical-edition-readiness')
    def edition_readiness(aid, eid):
        if not authenticated():
            return jsonify(error='Sign in to inspect saved financial dates.'), 403
        try:
            selection = bounded_selection({'baseline'})
            from historical_valuation_review import _baseline
            _baseline(selection['baseline'])
            from historical_valuation_candidate_context import edition_readiness_from_storage
            with get_db() as (_, cur):
                result = edition_readiness_from_storage(cur, aid, eid, selection['baseline'])
            return jsonify(result)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            return jsonify(error='Saved financial dates could not be checked for this edition. Reload its evidence; no retrieval or research was started.'), 409

    @bp.get('/api/research/assignments/<aid>/onepager-editions/<eid>/historical-readiness')
    def batch_readiness(aid, eid):
        if not authenticated():
            return jsonify(error='Sign in to inspect historical readiness.'), 403
        try:
            selection = bounded_selection({'baseline', 'dates'})
            from historical_valuation_readiness import validate_dates
            from historical_valuation_review import _baseline
            validate_dates(selection['dates'])
            _baseline(selection['baseline'])
            from historical_valuation_candidate_context import readiness_from_storage
            with get_db() as (_, cur):
                result = readiness_from_storage(cur, aid, eid, selection['dates'], selection['baseline'])
            return jsonify(result)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            return jsonify(error='Historical readiness could not be verified for this frozen edition. Reload its evidence; no retrieval or research was started.'), 409

    @bp.get('/api/research/assignments/<aid>/onepager-editions/<eid>/historical-plan')
    def selection_plan(aid, eid):
        if not authenticated():
            return jsonify(error='Sign in to inspect historical coverage.'), 403
        try:
            selection = bounded_selection({'baseline', 'date'})
            _date(selection['date'])
            from historical_valuation_review import _baseline
            _baseline(selection['baseline'])
            from historical_valuation_candidate_context import plan_from_storage
            with get_db() as (_, cur):
                result = plan_from_storage(cur, aid, eid, selection['date'], selection['baseline'])
            return jsonify(result)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            return jsonify(error='Retained coverage could not be checked for this frozen edition. Reload evidence or resolve the inventory bound; no retrieval or research was started.'), 409

    @bp.get('/api/research/assignments/<aid>/onepager-editions/<eid>/historical-candidate')
    def candidate_preview(aid, eid):
        if not authenticated():
            return jsonify(error='Sign in to inspect historical inputs.'), 403
        try:
            selection = bounded_selection({'baseline', 'observation'})
            from historical_valuation_candidate_context import assemble_from_storage
            with get_db() as (_, cur):
                result = assemble_from_storage(cur, aid, eid, [selection['observation']], selection['baseline'])
            return jsonify(result)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            return jsonify(error='Historical inputs could not be verified for this edition and selection. Reload the retained evidence; no retrieval or research was started.'), 409

    @bp.get('/api/research/assignments/<aid>/onepager-editions/<eid>/historical-retrievals')
    def retrieval_status(aid, eid):
        if not authenticated():
            return jsonify(error='Sign in to inspect historical retrieval.'), 403
        try:
            aid, eid = _identity(aid), _identity(eid)
            with get_db() as (_, cur):
                result = inspect_retrievals(cur, aid, eid)
            return jsonify(**result, assignmentId=aid, parentEditionId=eid)
        except (ValueError, TypeError, KeyError, AttributeError):
            return jsonify(error='Historical retrieval status is unavailable for this saved target. No retrieval or retry was started.'), 409

    @bp.route('/api/research/assignments/<aid>/onepager-editions/<eid>/historical-originals', methods=['GET', 'POST'])
    def ingest(aid, eid):
        if not authenticated():
            return jsonify(error='Sign in to retrieve historical evidence.'), 403
        # Credentials and provider entitlement are server configuration only.
        # Deliberately disabled unless the host explicitly supplies configuration.
        try:
            aid, eid = _identity(aid), _identity(eid)
            if request.method == 'GET':
                # Inventory remains readable if provider retrieval is disabled.
                # Never load source bodies simply to populate the review UI.
                with get_db() as (_, cur):
                    baseline, _ = load_context(cur, aid, eid)
                    cur.execute("SELECT to_regclass('historical_valuation_originals') AS name")
                    rows = []
                    if cur.fetchone()['name']:
                        cur.execute('''SELECT source_id,sha256,kind,retrieved_date,
                            octet_length(content) AS size FROM historical_valuation_originals
                            WHERE assignment_id=%s ORDER BY retrieved_date DESC,source_id LIMIT 61''', (aid,))
                        rows = cur.fetchall()
                    originals = [dict(id=r['source_id'],sha256=r['sha256'],kind=r['kind'],
                                      retrievedDate=str(r['retrieved_date']),bytes=r['size']) for r in rows[:60]]
                return jsonify(originals=originals,truncated=len(rows)>60,baseline=baseline,
                               valuationReady=False,caveat='Retained source inventory only; contents and valuation require separate verification.')
            config = ingestion_config() if ingestion_config else {}
            if not isinstance(config, dict) or not config.get('enabled'):
                return jsonify(error='Historical source retrieval is not configured.'), 503
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict) or set(payload) - {'kind', 'priceDate'}:
                raise ValueError('Choose a supported source and date; provider settings cannot be supplied here.')
            from historical_valuation_ingest import ingest_sec, ingest_prices
            if payload.get('kind') == 'sec_companyfacts' and 'priceDate' not in payload:
                fetch = lambda: ingest_sec(get_db, aid, eid, user_agent=config.get('sec_user_agent'), retrieval_id=request.headers.get('Idempotency-Key'))
            elif payload.get('kind') == 'daily_prices':
                if not config.get('compact_prices_enabled'):
                    return jsonify(error='Daily price retrieval requires configured provider access.'), 503
                # Invalid input never reaches the durable provider reservation.
                # Such a reservation otherwise looks like an unknown network outcome.
                if _date(payload.get('priceDate')) >= datetime.now(timezone.utc).date():
                    raise ValueError('Choose a prior completed trading date.')
                fetch = lambda: ingest_prices(get_db, aid, eid, payload.get('priceDate'), api_key=config.get('price_api_key'), retrieval_id=request.headers.get('Idempotency-Key'))
            else:
                raise ValueError('Unsupported historical source.')
            result = retrieve_once(get_db, aid, eid, request.headers.get('Idempotency-Key'), payload, fetch)
            return jsonify(original=result, status='evidence_retained', valuationReady=False), 200
        except (ValueError, TypeError, KeyError, AttributeError):
            # Never echo provider/configuration exception strings into a response.
            return jsonify(error='Historical evidence was not verified. Check the target, source configuration and date coverage before another retrieval. No automatic retry was made.'), 409

    @bp.get('/api/research/assignments/<aid>/onepager-editions/<eid>/historical-reviews')
    def review_inventory(aid, eid):
        if not authenticated():
            return jsonify(error='Sign in to inspect saved historical reviews.'), 403
        try:
            aid, eid = _identity(aid), _identity(eid)
            with get_db() as (_, cur):
                baseline, _ = load_context(cur, aid, eid)
                cur.execute("SELECT to_regclass('historical_valuation_reviews') AS name")
                rows = []
                if cur.fetchone()['name']:
                    cur.execute('''SELECT id,review_digest,created_at,
                        snapshot->'payload'->'baseline' AS baseline
                        FROM historical_valuation_reviews
                        WHERE assignment_id=%s AND parent_edition_id=%s
                        ORDER BY created_at DESC,id LIMIT 61''', (aid,eid))
                    rows = cur.fetchall()
                reviews = [{'id':str(r['id']), 'reviewDigest':r['review_digest'],
                            'createdAt':r['created_at'].isoformat(), 'status':'pending',
                            'baselineChanged':r['baseline'] != baseline,
                            'requiresEvidenceRecheck':True} for r in rows[:60]]
            return jsonify(reviews=reviews, truncated=len(rows)>60, valuationReady=False, baseline=baseline,
                           assignmentId=aid, parentEditionId=eid,
                           caveat='Saved review index only. Open a version to recheck its evidence; no approval is granted.')
        except (ValueError, TypeError, KeyError, AttributeError):
            return jsonify(error='Saved historical reviews are unavailable for this target.'), 409

    @bp.route('/api/research/assignments/<aid>/onepager-editions/<eid>/historical-reviews/<rid>', methods=['GET', 'POST'])
    def review(aid, eid, rid):
        if not authenticated():
            return jsonify(error='Sign in to review historical evidence.'), 403
        try:
            aid, eid, rid = _identity(aid), _identity(eid), _identity(rid)
            writing = request.method == 'POST'
            proposal = request.get_json(silent=True) if writing else None
            if writing:
                if not isinstance(proposal, dict) or len(json.dumps(proposal, allow_nan=False).encode()) > 500000:
                    raise ValueError('Invalid or oversized historical proposal.')
                refs = references(proposal)
            with get_db(commit=writing) as (_, cur):
                baseline, identity = load_context(cur, aid, eid)
                if not writing:
                    cur.execute("SELECT to_regclass('historical_valuation_reviews') AS name")
                    if not cur.fetchone()['name']:
                        return jsonify(error='Saved historical review not found.'), 404
                    cur.execute('SELECT snapshot FROM historical_valuation_reviews WHERE id=%s AND assignment_id=%s AND parent_edition_id=%s FOR SHARE', (rid, aid, eid))
                    saved = cur.fetchone()
                    if not saved:
                        return jsonify(error='Saved historical review not found.'), 404
                    refs = references(saved['snapshot']['payload']['proposal'])
                originals, dates = read_originals(cur, refs, assignment_id=aid)
                context = {**identity, 'price_retrieval_dates': dates}
                if writing:
                    result = save_pending(cur, rid, proposal, originals, baseline, **context)
                    return jsonify(result), 200 if result['reused'] else 201
                snapshot = read_pending(cur, rid, originals, baseline, **context)
                return jsonify(id=rid, status='pending', snapshot=snapshot,
                               caveat='Pending diagnostics only. Issuer, share-class, corporate-action and filing completeness checks remain unresolved; no valuation approval is granted.')
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return jsonify(error=str(exc)), 409
    return bp
