"""Dependency-free assignment validation shared by the Mac and server."""
import re
from datetime import date,datetime
from zoneinfo import ZoneInfo

DEFAULT_OUTPUTS=('summary_note','stock_summary','thesis','visual')
OUTPUTS=DEFAULT_OUTPUTS+('meeting_prep',)

def plan(data):
    if not isinstance(data,dict):raise ValueError('Expected a research assignment.')
    ticker=data.get('ticker','')
    if not isinstance(ticker,str):raise ValueError('Enter one ticker.')
    ticker=ticker.strip().upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,14}',ticker):raise ValueError('Enter one ticker.')
    try:since=date.fromisoformat(data['since']);until=date.fromisoformat(data['until'])
    except (ValueError,TypeError,KeyError):raise ValueError('Choose a start and end date.')
    if since>until or (until-since).days>=365 or until>datetime.now(ZoneInfo('America/New_York')).date():raise ValueError('Choose a past/present range of 1–365 days.')
    outputs=data.get('outputs',list(DEFAULT_OUTPUTS));horizon=data.get('horizon','12–24 months');instruction=data.get('instruction','')
    if not isinstance(outputs,list) or not outputs or any(not isinstance(x,str) or x not in OUTPUTS for x in outputs) or len(set(outputs))!=len(outputs):raise ValueError('Choose at least one supported output.')
    if not isinstance(horizon,str) or not 1<=len(horizon.strip())<=100 or not isinstance(instruction,str) or len(instruction)>1800:raise ValueError('Invalid horizon or research instructions.')
    supplement=data.get('visualFundamentals',False)
    if not isinstance(supplement,bool):raise ValueError('Invalid public fundamentals choice.')
    result=dict(ticker=ticker,since=since.isoformat(),until=until.isoformat(),outputs=sorted(outputs),horizon=horizon.strip(),instruction=instruction.strip())
    if 'meeting_prep' in outputs:
        meeting=data.get('meeting')
        if not isinstance(meeting,dict) or set(meeting)!={'date','type'}:
            raise ValueError('Choose the meeting date and type for the shared research pack.')
        try:meeting_date=date.fromisoformat(meeting['date']).isoformat()
        except (ValueError,TypeError,KeyError):raise ValueError('Choose a valid meeting date.')
        if meeting['type'] not in ('management','expert','conference','other'):
            raise ValueError('Choose a valid meeting type.')
        result['meeting']={'date':meeting_date,'type':meeting['type']}
    elif data.get('meeting') is not None:
        raise ValueError('Select meeting preparation before supplying meeting settings.')
    if supplement:result['visualFundamentals']=True
    if 'origin' in data:
        from research_assignment_origin import validate
        result['origin']=validate(data['origin'],ticker,outputs)
    if 'labSelection' in data:
        selection=data['labSelection']
        if (not isinstance(selection,dict) or set(selection)!={'task','revision','digest'}
                or selection['task'] not in ('thesis','note') or type(selection['revision']) is not int
                or selection['revision']<1 or not isinstance(selection['digest'],str)
                or not re.fullmatch(r'[0-9a-f]{64}',selection['digest'])):
            raise ValueError('Check the saved Lab selection before starting.')
        if selection['task']=='note' and 'thesis' in outputs:
            raise ValueError('Use the separate thesis selection to prepare thesis proposals.')
        result['labSelection']=dict(selection)
    return result
