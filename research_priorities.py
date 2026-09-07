"""User-reported exposures for research ordering, never trading instructions."""
import json
import math
import re
from datetime import date
from flask import Blueprint,jsonify,request
KEY='research_priorities_v1'


def validate(value,today=None):
    if not isinstance(value,dict):raise ValueError('Portfolio settings are required.')
    try:asof=date.fromisoformat(value.get('asOf',''))
    except (ValueError,TypeError):raise ValueError('Set the holdings as-of date.')
    if asof>(today or date.today()):raise ValueError('Holdings cannot be dated in the future.')
    rows=value.get('positions')
    if not isinstance(rows,list) or len(rows)>200:raise ValueError('Use up to 200 positions.')
    seen=set();positions=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Invalid position.')
        tk=row.get('ticker');weight=row.get('weightPct')
        if not isinstance(tk,str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}',tk) or tk in seen:raise ValueError('Use distinct uppercase tickers.')
        if type(weight) not in (int,float) or not math.isfinite(weight) or not -100<=weight<=100:raise ValueError('Each signed weight must be between -100% and 100%.')
        seen.add(tk);positions.append({'ticker':tk,'weightPct':weight})
    return {'asOf':asof.isoformat(),'positions':positions,'source':'user_reported','staleAfterDays':30}


def create_blueprint(get_db):
    bp=Blueprint('research_priorities',__name__)
    @bp.route('/api/research/priorities',methods=['GET','POST'])
    def preferences():
        if request.method=='POST':
            try:value=validate(request.get_json(silent=True))
            except ValueError as e:return jsonify(error=str(e)),400
            with get_db(commit=True) as (_,cur):cur.execute('INSERT INTO app_settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()',(KEY,json.dumps(value)))
        else:
            with get_db() as (_,cur):cur.execute('SELECT value FROM app_settings WHERE key=%s',(KEY,));row=cur.fetchone()
            value=json.loads(row['value']) if row and isinstance(row['value'],str) else row['value'] if row else None
        r=jsonify(profile=value);r.headers['Cache-Control']='no-store';return r
    return bp
