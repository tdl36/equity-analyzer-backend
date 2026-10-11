"""Read-only public fundamentals supplement. No models, approvals or research rewrites."""
import hashlib
import math
import re
from datetime import date, datetime, timezone
import requests
from sec_edgar import USER_AGENT

METRICS={'revenue':('Revenues','RevenueFromContractWithCustomerExcludingAssessedTax','SalesRevenueNet'),
 'operatingIncome':('OperatingIncomeLoss',),'eps':('EarningsPerShareDiluted',),
 'operatingCashFlow':('NetCashProvidedByUsedInOperatingActivities',),
 'capex':('PaymentsToAcquirePropertyPlantAndEquipment',),'netIncome':('NetIncomeLoss',)}

def annuals(raw, as_of):
    """Latest filed annual observations, same period and accession for arithmetic.

    Non-calendar fiscal years are supported by start/end duration. Quarterly,
    future-filed, non-10K and conflicting same-filing values are not charted.
    """
    result={};facts=raw.get('facts',{}).get('us-gaap',{})
    for metric,tags in METRICS.items():
        unit='USD/shares' if metric=='eps' else 'USD'
        for tag in tags:
            candidates={}
            for f in facts.get(tag,{}).get('units',{}).get(unit,[]):
                try:
                    start=date.fromisoformat(f['start']);end=date.fromisoformat(f['end'])
                    if f['filed']>as_of or f['end']>as_of or f['form'] not in ('10-K','10-K/A') or not 330<=(end-start).days<=380:continue
                    val=float(f['val'])
                    if not math.isfinite(val):continue
                except (ValueError,KeyError,TypeError):continue
                candidates.setdefault((f['start'],f['end']),[]).append(f)
            for period,fs in candidates.items():
                latest=max(f['filed'] for f in fs);same=[f for f in fs if f['filed']==latest]
                if len({float(f['val']) for f in same})!=1:continue
                f=sorted(same,key=lambda x:x['accn'])[-1]
                row=result.setdefault(period,{'start':period[0],'end':period[1],'year':period[1][:4]})
                if metric not in row:row[metric]={'value':f['val'],'unit':unit,'tag':tag,'filed':f['filed'],'accession':f['accn']}
    ends=[r['end'] for r in result.values()]
    rows=sorted([r for r in result.values() if ends.count(r['end'])==1],key=lambda r:r['end'])[-5:]
    for row in rows:
        for target,a,b,scale in [('operatingMargin','operatingIncome','revenue',100),('freeCashFlow','operatingCashFlow','capex',1)]:
            x,y=row.get(a),row.get(b)
            if x and y and x['accession']==y['accession'] and (target!='operatingMargin' or y['value']>0):
                value=(x['value']/y['value']*scale) if target=='operatingMargin' else x['value']-y['value']
                row[target]={'value':value,'unit':'%' if target=='operatingMargin' else 'USD','calculation':f'{a} / {b} × 100' if target=='operatingMargin' else f'{a} − {b}','accession':x['accession'],'filed':x['filed']}
    return rows

def positive_number(value):
    try:
        number=float(value)
        return number if math.isfinite(number) and number>0 and not isinstance(value,bool) else None
    except (TypeError,ValueError):return None

def public_json(url):
    r=requests.get(url,headers={'User-Agent':USER_AGENT},timeout=25,allow_redirects=False)
    r.raise_for_status()
    if len(r.content)>15000000:raise ValueError('Public data response exceeds the bounded size.')
    return r.json(),hashlib.sha256(r.content).hexdigest()

def collect(ticker):
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,14}',ticker):raise ValueError('Invalid ticker.')
    now=datetime.now(timezone.utc);asof=now.date().isoformat()
    mapping,_=public_json('https://www.sec.gov/files/company_tickers.json')
    matches=[v for v in mapping.values() if v.get('ticker')==ticker]
    if len(matches)!=1:raise ValueError('A unique SEC issuer was not found; public fundamentals need review.')
    cik=int(matches[0]['cik_str']);url=f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'
    raw,sha=public_json(url)
    if int(raw.get('cik',-1))!=cik:raise ValueError('SEC issuer identity mismatch.')
    rows=annuals(raw,asof)
    if len([r for r in rows if 'revenue' in r])<3:raise ValueError('At least three comparable annual revenue observations are required.')
    out={'version':2,'ticker':ticker,'company':raw.get('entityName'),'cik':cik,'retrievedAt':now.isoformat(),'asOf':asof,'annuals':rows,'sourceUrl':url,'sourceHash':sha,'quote':None,'gaps':[]}
    try:out['quote']=quote(ticker,now)
    except Exception:out['gaps'].append('Current market quote unavailable or not validated; no current valuation inferred.')
    from visual_business_profiles import PROFILES
    profile=PROFILES.get(ticker,{})
    out['peers']=[]
    for symbol,name,role in profile.get('peers',[]):
        try:
            peer=quote(symbol,now)
            # Comparable market snapshots must come from the same trading date.
            if not out['quote'] or peer['observedAt'][:10]!=out['quote']['observedAt'][:10]:raise ValueError('Quote dates differ.')
            out['peers'].append(dict(ticker=symbol,name=name,role=role,quote=peer))
        except Exception:out['gaps'].append(f'{symbol} peer quote unavailable or not date-matched.')
    try:add_segments(out)
    except Exception:out['gaps'].append('Segment filing data could not be validated.')
    return out

def segment_facts(content, start, end, cik):
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(content,'html.parser');contexts={};result={}
    from visual_business_profiles import PROFILES
    profile=next((p for p in PROFILES.values() if p.get('cik')==cik),{})
    usd_units=set()
    for unit in soup.find_all(lambda t:t.name.endswith(':unit')):
        measures=unit.find_all(lambda t:t.name.endswith(':measure'))
        if len(measures)==1 and measures[0].text.strip()=='iso4217:USD' and not unit.find(lambda t:t.name.endswith(':divide')):
            usd_units.add(unit.get('id'))
    names={'Unitedhealthcare':'UnitedHealthcare','Optumhealth':'Optum Health','Optuminsight':'Optum Insight','Optumrx':'Optum Rx'}
    for c in soup.find_all(lambda t:t.name.endswith(':context')):
        ident=c.find(lambda t:t.name.endswith(':identifier'))
        a=c.find(lambda t:t.name.endswith(':startdate'));b=c.find(lambda t:t.name.endswith(':enddate'))
        if not ident or ident.text.lstrip('0')!=str(cik) or not a or not b or a.text!=start or b.text!=end:continue
        dims={m.get('dimension'):m.text for m in c.find_all(lambda t:t.name.endswith(':explicitmember'))}
        member=dims.pop('us-gaap:StatementBusinessSegmentsAxis',None)
        if not member or dims not in ({},{'srt:ConsolidationItemsAxis':'us-gaap:OperatingSegmentsMember'}):continue
        if profile.get('segmentScope')=='operating' and dims!={'srt:ConsolidationItemsAxis':'us-gaap:OperatingSegmentsMember'}:continue
        raw=member.split(':')[-1].removesuffix('Member')
        if any(x in raw.lower() for x in ('total','elimination','corporate','other')):continue
        name=profile.get('segmentMembers',{}).get(member) or names.get(raw,re.sub(r'(?<=[a-z])(?=[A-Z])',' ',raw))
        contexts[c['id']]=(member,name)
    for x in soup.find_all('ix:nonfraction'):
        if x.get('contextref') not in contexts or x.get('name') not in {'us-gaap:'+tag for tag in METRICS['revenue']} or x.get('format') not in ('ixt:num-dot-decimal','ixt:numdotdecimal') or x.get('unitref') not in usd_units:continue
        try:
            value=float(x.text.replace(',',''))*10**int(x.get('scale','0'))
            if x.get('sign')=='-':value=-value
            if not math.isfinite(value) or value<=0:continue
        except (ValueError,TypeError,OverflowError):continue
        member,name=contexts[x['contextref']]
        key=name if member in profile.get('segmentMembers',{}) else member
        result.setdefault(key,{'name':name,'values':set()})['values'].add(value)
    return [{'name':r['name'],'revenue':next(iter(r['values'])),'year':end[:4],'basis':'Segment revenues, including intersegment activity where reported'} for r in result.values() if len(r['values'])==1][:6]

def add_segments(out):
    cik=out['cik'];sub,_=public_json(f'https://data.sec.gov/submissions/CIK{cik:010d}.json')
    recent=sub.get('filings',{}).get('recent',{});latest=out['annuals'][-1]
    for i,form in enumerate(recent.get('form',[])):
        if form=='10-K' and recent['reportDate'][i]==latest['end'] and recent['filingDate'][i]<=out['asOf']:
            acc=recent['accessionNumber'][i];doc=recent['primaryDocument'][i]
            if not re.fullmatch(r'[A-Za-z0-9._-]+',doc) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}',acc):break
            url=f'https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace("-","")}/{doc}'
            r=requests.get(url,headers={'User-Agent':USER_AGENT},timeout=25,allow_redirects=False);r.raise_for_status()
            if len(r.content)>15000000:break
            out['segments']=segment_facts(r.content,latest['start'],latest['end'],cik)
            if not out['segments']:out['gaps'].append('No unambiguous segment revenue mapping was found in the annual filing.')
            out['segmentSource']={'url':url,'sha256':hashlib.sha256(r.content).hexdigest(),'filed':recent['filingDate'][i]}
            return


def quote(ticker,now):
    import yfinance as yf
    info=yf.Ticker(ticker).get_info()
    stamp=info.get('regularMarketTime');price=positive_number(info.get('regularMarketPrice'))
    if info.get('symbol')!=ticker or info.get('currency')!='USD' or not stamp or not price:raise ValueError('Invalid quote identity, currency or price.')
    qt=datetime.fromtimestamp(stamp,timezone.utc)
    if not 0<=(now-qt).total_seconds()<=7*86400:raise ValueError('Stale or future quote.')
    return dict(ticker=ticker,price=price,currency='USD',observedAt=qt.isoformat(),retrievedAt=now.isoformat(),sourceUrl=f'https://finance.yahoo.com/quote/{ticker}/',provider='Yahoo Finance',basis='Regular market quote; provider-defined multiples',
        **{k:positive_number(info.get(k)) for k in ['marketCap','trailingPE','forwardPE','forwardEps']},
        **{k:info.get(k) for k in ['sector','industry','exchange']})
