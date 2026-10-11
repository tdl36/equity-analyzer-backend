"""Read-only job-scoped accounting evidence. No replay or billing inference."""
def read_usage(cur, run_id, feature):
    cur.execute("""SELECT created_at, provider, model, input_tokens, output_tokens,
                          cost_usd, attempt, detail->>'stage' AS stage
                   FROM llm_usage WHERE detail->>'runId'=%s AND feature=%s
                   ORDER BY created_at, id LIMIT 101""", (run_id, feature))
    rows = list(cur.fetchall() or [])
    return {'records': [{
        'recordedAt': str(r['created_at']), 'provider': r['provider'], 'model': r['model'],
        'stage': r['stage'], 'inputTokens': r['input_tokens'], 'outputTokens': r['output_tokens'],
        'estimatedCostUsd': float(r['cost_usd']) if r['cost_usd'] is not None else None,
        'attempt': r['attempt'],
    } for r in rows[:100]], 'truncated': len(rows)>100,
    'limitation': 'Charlie usage records are estimated accounting evidence, not a provider invoice or proof that an interrupted call is safe to retry. Missing records do not establish zero usage.'}
