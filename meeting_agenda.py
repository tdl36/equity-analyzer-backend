"""Bounded, read-only view of dated meeting preparation; not a calendar sync."""
from datetime import timedelta
from zoneinfo import ZoneInfo


def upcoming(cur,now):
    start=now.astimezone(ZoneInfo('America/New_York')).date()
    end=start+timedelta(days=6)
    cur.execute("SELECT to_regclass('mp_meetings') AS name")
    if not cur.fetchone()['name']:raise RuntimeError('Meeting preparation inventory unavailable')
    cur.execute('''SELECT m.id,c.ticker,m.meeting_date,m.meeting_type,
        EXISTS(SELECT 1 FROM mp_question_sets q WHERE q.meeting_id=m.id AND q.status='ready') AS prepared
        FROM mp_meetings m JOIN mp_companies c ON c.id=m.company_id
        WHERE m.meeting_date BETWEEN %s AND %s
        AND COALESCE(m.status,'draft') NOT IN ('completed','cancelled','archived')
        ORDER BY m.meeting_date,c.ticker,m.id LIMIT 51''',(start,end))
    rows=list(cur.fetchall())
    return {'items':[{'id':r['id'],'ticker':r['ticker'],'date':str(r['meeting_date']),
                    'type':r['meeting_type'],'prepared':bool(r['prepared'])} for r in rows[:50]],
            'from':str(start),'through':str(end),'truncated':len(rows)>50}
