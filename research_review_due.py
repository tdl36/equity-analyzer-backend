"""Read-only reminders for analyst-recorded issue review dates; no review mutation."""
from datetime import date
from zoneinfo import ZoneInfo


def issue_reviews(cur, due_through=None, limit=200):
    """Shared issue selection for Activity and the existing research due queue."""
    cur.execute("SELECT to_regclass('research_decisions') AS name")
    if not cur.fetchone()['name']:return [],False
    cutoff=" AND d.body->>'reviewDate'<=%s" if due_through else ''
    cur.execute(r'''SELECT d.id,d.ticker,d.revision,d.created_at,d.body,
        COALESCE(NULLIF(d.body->>'issue',''),d.body->>'decision') AS issue,d.body->>'reviewDate' AS due
        FROM research_decisions d WHERE d.body->>'reviewDate' ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
        AND NOT EXISTS(SELECT 1 FROM research_decisions n
            WHERE n.ticker=d.ticker AND n.body->>'supersedes'=d.id)'''+cutoff+
        " ORDER BY d.body->>'reviewDate',d.created_at LIMIT %s",
        ((due_through,) if due_through else ())+ (limit+1,))
    rows=list(cur.fetchall());valid=[]
    for row in rows[:limit]:
        try:date.fromisoformat(row['due'])
        except (ValueError,TypeError):continue
        if row.get('issue'):valid.append(dict(row))
    return valid,len(rows)>limit


def due_items(cur, now):
    today=now.astimezone(ZoneInfo('America/New_York')).date()
    rows,more=issue_reviews(cur,today.isoformat());items=[]
    from activity_snapshot import stamp
    for r in rows:
        try:due=date.fromisoformat(r['due'])
        except (ValueError,TypeError):continue
        if due>today:continue
        items.append(dict(id='decision:'+str(r['id']),jobId=str(r['id']),ticker=r['ticker'],
            title='Research review due',workflow='research_decision',status='needs_review',bucket='attention',
            step=('Review due today' if due==today else 'Review overdue since '+due.isoformat())+' · '+r['issue'][:200],
            createdAt=stamp(r['created_at']),updatedAt=stamp(r['created_at']),dueDate=due.isoformat(),
            view='stockanalysis',error='',stale=False,quietMinutes=None,
            warning='Based on your recorded review date (New York). Opening this item does not complete the review.'))
    return items,more
