"""Bounded read-only assignment archive; no coordinator wake or research bodies."""
import base64
import json
import re
import uuid
from datetime import datetime


def cursor_for(row, ticker):
    stamp=row['created_at']
    return base64.urlsafe_b64encode(json.dumps([stamp.isoformat(),str(row['id']),ticker],separators=(',',':')).encode()).decode()


def read_page(cur, ticker='', cursor=None):
    if not isinstance(ticker,str) or (ticker and not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,29}',ticker)):
        raise ValueError('Invalid company filter.')
    clauses=["stage='research_assignment'"];params=[]
    if ticker:clauses.append('ticker=%s');params.append(ticker)
    if cursor is not None:
        try:
            if not isinstance(cursor,str) or not 1<=len(cursor)<=512:raise ValueError()
            value=json.loads(base64.b64decode(cursor,altchars=b'-_',validate=True))
            if not isinstance(value,list) or len(value)!=3 or value[2]!=ticker:raise ValueError()
            stamp=datetime.fromisoformat(value[0]);ident=str(uuid.UUID(value[1]))
            if ident!=value[1]:raise ValueError()
        except (ValueError,TypeError,AttributeError,UnicodeError):
            raise ValueError('Invalid archive cursor.') from None
        clauses.append('(created_at,id)<(%s,%s)');params.extend([stamp,ident])
    cur.execute('SELECT id,ticker,status,created_at,updated_at FROM mp_jobs WHERE '+
                ' AND '.join(clauses)+' ORDER BY created_at DESC,id DESC LIMIT 31',tuple(params))
    rows=cur.fetchall()
    return {'assignments':[dict(row) for row in rows[:30]],
            'nextCursor':cursor_for(rows[29],ticker) if len(rows)>30 else None,
            'readOnly':True}
