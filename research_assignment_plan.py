"""Dependency-free assignment validation shared by the Mac and server."""
import re
from datetime import date,datetime
from zoneinfo import ZoneInfo

OUTPUTS=('summary_note','stock_summary','thesis','visual')

def plan(data):
    if not isinstance(data,dict):raise ValueError('Expected a research assignment.')
    ticker=data.get('ticker','')
    if not isinstance(ticker,str):raise ValueError('Enter one ticker.')
    ticker=ticker.strip().upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,14}',ticker):raise ValueError('Enter one ticker.')
    try:since=date.fromisoformat(data['since']);until=date.fromisoformat(data['until'])
    except (ValueError,TypeError,KeyError):raise ValueError('Choose a start and end date.')
    if since>until or (until-since).days>=365 or until>datetime.now(ZoneInfo('America/New_York')).date():raise ValueError('Choose a past/present range of 1–365 days.')
    outputs=data.get('outputs',list(OUTPUTS));horizon=data.get('horizon','12–24 months');instruction=data.get('instruction','')
    if not isinstance(outputs,list) or not outputs or any(not isinstance(x,str) or x not in OUTPUTS for x in outputs) or len(set(outputs))!=len(outputs):raise ValueError('Choose at least one supported output.')
    if not isinstance(horizon,str) or not 1<=len(horizon.strip())<=100 or not isinstance(instruction,str) or len(instruction)>1800:raise ValueError('Invalid horizon or research instructions.')
    return dict(ticker=ticker,since=since.isoformat(),until=until.isoformat(),outputs=sorted(outputs),horizon=horizon.strip(),instruction=instruction.strip())
