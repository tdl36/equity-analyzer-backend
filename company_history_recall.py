"""Read-only recall of recorded answers and cached original-document passages.

These are saved records, not a fresh download or independent verification.
"""
import hashlib


def available(cur, name):
    cur.execute('SELECT to_regclass(%s) AS name',(name,))
    return bool(cur.fetchone()['name'])


def recall(cur,ticker,terms,as_of,include_sources=False):
    entries=[]
    receipt={'method':'literal_terms_then_recency','terms':terms,'asOf':as_of,
        'answersOmitted':False,'sourcesOmitted':False,'sourcesRequested':include_sources,
        'limitations':['Answers are analyst-recorded notes, not verified verbatim management statements.',
                      'Matching can miss relevant history. Source excerpts are partial cached extraction, not freshly verified originals.']}
    if not all(available(cur,t) for t in ('mp_companies','mp_meetings')):
        receipt['limitations'].append('Meeting tables unavailable.');return entries,receipt
    if available(cur,'mp_past_questions'):
        cur.execute('''SELECT q.id,q.meeting_id,q.question,q.response_notes,q.status,m.meeting_date,
            (SELECT count(*) FROM unnest(%s::text[]) term WHERE term=ANY(regexp_split_to_array(
              lower(concat_ws(' ',q.question,q.response_notes,q.topic)), '[^a-z0-9-]+'))) AS relevance
            FROM mp_past_questions q JOIN mp_companies c ON c.id=q.company_id
            JOIN mp_meetings m ON m.id=q.meeting_id AND m.company_id=c.id
            WHERE upper(c.ticker)=%s AND m.meeting_date<=%s
              AND q.status IN ('answered','resolved') AND length(trim(coalesce(q.response_notes,'')))>0
            ORDER BY relevance DESC,m.meeting_date DESC,q.id DESC LIMIT 9''',(terms,ticker,as_of))
        rows=cur.fetchall();receipt['answersOmitted']=len(rows)>8
        for row in rows[:8]:
            body={k:str(row[k]) if k=='meeting_date' else row[k] for k in ('id','meeting_id','question','response_notes','status','meeting_date')}
            entries.append({'kind':'meeting_answer','status':'analyst_recorded_answer_not_reverified','body':body})
    if include_sources and terms and available(cur,'mp_documents'):
        # Each excerpt is an exact substring of stored extraction with its offset.
        # Do not load original binary blobs or claim a page number we do not retain.
        cur.execute('''SELECT d.id,d.meeting_id,d.filename,d.doc_date,
            encode(sha256(convert_to(d.extracted_text,'UTF8')),'hex') AS extraction_digest,
            length(d.extracted_text) AS total_characters,
            hit.position AS match_position,
            substring(d.extracted_text FROM greatest(1,hit.position-300) FOR 1800) AS passage
            FROM mp_documents d JOIN mp_meetings m ON m.id=d.meeting_id
            JOIN mp_companies c ON c.id=m.company_id
            CROSS JOIN LATERAL (SELECT min(nullif(strpos(lower(d.extracted_text),term),0)) AS position
                FROM unnest(%s::text[]) term) hit
            WHERE upper(c.ticker)=%s AND m.meeting_date<=%s AND hit.position IS NOT NULL
              AND coalesce(d.doc_date,'')<=%s
            ORDER BY m.meeting_date DESC,d.id DESC LIMIT 7''',(terms,ticker,as_of,as_of))
        rows=cur.fetchall();receipt['sourcesOmitted']=len(rows)>6
        seen=set()
        for row in rows[:6]:
            if row['extraction_digest'] in seen:continue
            seen.add(row['extraction_digest'])
            body=dict(row);body['startCharacter']=max(1,row['match_position']-300)
            body['passageSha256']=hashlib.sha256(row['passage'].encode()).hexdigest()
            body['textUrl']=f"/api/mp/meetings/{row['meeting_id']}/documents/{row['id']}/text"
            body['priorIssueReviews']=[]
            if available(cur,'research_decisions'):
                cur.execute('''SELECT d.id,d.revision,d.body->>'issue' AS issue,d.body->>'disposition' AS disposition,
                    d.body->>'rationale' AS rationale,d.body->>'revisitWhen' AS revisit_when,d.body->>'reviewDate' AS review_date
                    FROM research_decisions d WHERE d.ticker=%s AND d.body->>'issueId' IS NOT NULL
                    AND EXISTS(SELECT 1 FROM jsonb_array_elements(coalesce(d.body->'reviewedSources','[]'::jsonb)) s
                        WHERE s->>'extraction_hash'=%s)
                    AND NOT EXISTS(SELECT 1 FROM research_decisions n WHERE n.ticker=d.ticker AND n.body->>'supersedes'=d.id)
                    ORDER BY d.revision DESC LIMIT 5''',(ticker,row['extraction_digest']))
                body['priorIssueReviews']=[dict(r) for r in cur.fetchall()]
            body['reviewScope']='Exact saved extraction matches only; no alert suppressed. New or changed evidence requires fresh assessment.'
            entries.append({'kind':'saved_source_excerpt','status':'cached_extraction_not_reverified','body':body})
    return entries,receipt
