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

def annuals(raw, as_of, issues=None):
    """Latest filed annual observations across supported tags; never mix vintages.

    Conflicting latest-filing values are withheld, rather than falling back to an
    older value. Arithmetic requires identical periods and filing accessions.
    """
    result={};facts=raw.get('facts',{}).get('us-gaap',{});issues=issues if issues is not None else []
    cutoff=date.fromisoformat(as_of);pending=[]
    for metric,tags in METRICS.items():
        unit='USD/shares' if metric=='eps' else 'USD';candidates={}
        for tag in tags:
            for f in facts.get(tag,{}).get('units',{}).get(unit,[]):
                try:
                    start=date.fromisoformat(f['start']);end=date.fromisoformat(f['end']);filed=date.fromisoformat(f['filed'])
                    if filed>cutoff or end>cutoff or filed<end or f['form'] not in ('10-K','10-K/A') or not 330<=(end-start).days<=380:continue
                    if isinstance(f['val'],bool) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}',f['accn']):continue
                    val=float(f['val'])
                    if not math.isfinite(val):continue
                except (ValueError,KeyError,TypeError,OverflowError):continue
                candidates.setdefault((f['start'],f['end']),[]).append((tag,f,val))
        for period,fs in candidates.items():
            latest=max(f['filed'] for _,f,_ in fs);same=[x for x in fs if x[1]['filed']==latest]
            if len({v for _,_,v in same})!=1:
                pending.append((period[1],f'{metric} for {period[1]} withheld: conflicting values in the latest filed observations ({latest}).'))
                continue
            tag,f,value=min(same,key=lambda x:(tags.index(x[0]),x[1]['accn']))
            row=result.setdefault(period,{'start':period[0],'end':period[1],'year':period[1][:4]})
            row[metric]={'value':value,'unit':unit,'tag':tag,'filed':f['filed'],'accession':f['accn'],
                         'form':f['form'],'observationBasis':'Latest filed observation available at retrieval; may include restatements.'}
    ends=[r['end'] for r in result.values()]
    for end in sorted(set(ends)):
        if ends.count(end)>1:issues.append(f'Annual history ending {end} withheld: more than one annual start date.')
    rows=sorted([r for r in result.values() if ends.count(r['end'])==1],key=lambda r:r['end'])[-5:]
    recent_ends={r['end'] for r in rows} or set(sorted({end for end,_ in pending})[-5:])
    issues.extend(message for end,message in pending if end in recent_ends)
    for row in rows:
        for target,a,b,scale in [('operatingMargin','operatingIncome','revenue',100),('freeCashFlow','operatingCashFlow','capex',1)]:
            x,y=row.get(a),row.get(b)
            if x and y and x['accession']==y['accession'] and (target!='operatingMargin' or y['value']>0):
                value=(x['value']/y['value']*scale) if target=='operatingMargin' else x['value']-y['value']
                if not math.isfinite(value):continue
                row[target]={'value':value,'unit':'%' if target=='operatingMargin' else 'USD','calculation':f'{a} / {b} × 100' if target=='operatingMargin' else f'{a} − {b}','accession':x['accession'],'filed':x['filed']}
            elif x and y:
                issues.append(f'{target} for {row["end"]} withheld: inputs have different filings or an invalid denominator.')
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
    issues=[];rows=annuals(raw,asof,issues)
    if len([r for r in rows if 'revenue' in r])<3:raise ValueError('At least three comparable annual revenue observations are required.')
    out={'version':3,'ticker':ticker,'company':raw.get('entityName'),'cik':cik,'retrievedAt':now.isoformat(),'asOf':asof,'annuals':rows,'sourceUrl':url,'sourceHash':sha,'quote':None,'gaps':[], 'financialWarnings':issues,
         'historyBasis':'Latest filed annual observations as of retrieval, including restatements; not a historical point-in-time backtest.'}
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
