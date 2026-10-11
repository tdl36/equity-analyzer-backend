"""Human-readable, source-linked calculation register for a frozen visual snapshot."""
import html
import math
import re

LABELS={'revenue':'Revenue','operatingIncome':'Operating income','eps':'Diluted EPS',
        'operatingCashFlow':'Operating cash flow','capex':'Capital expenditure','netIncome':'Net income',
        'operatingMargin':'Operating margin','freeCashFlow':'Free cash flow'}


def register(snapshot):
    esc=lambda value:html.escape(str(value if value is not None else ''),quote=True)
    notes=list(snapshot.get('financialWarnings') or [])+list(snapshot.get('gaps') or [])
    out='<h3>Financial calculation register</h3><p>'+esc(snapshot.get('historyBasis') or 'Saved annual observations; check each filing date. Historical values may include later restatements.')+'</p>'
    out+='<p>Retrieved '+esc(snapshot.get('retrievedAt','Unavailable'))+'. Amounts below are unscaled source units. FCF is consolidated cash flow less reported capex, not necessarily industrial or distributable cash flow.</p>'
    if notes:out+='<ul>'+''.join('<li>'+esc(note)+'</li>' for note in notes)+'</ul>'
    out+='<table><thead><tr><th>Fiscal period</th><th>Metric / exact value</th><th>Source tag or calculation</th><th>Filed / source</th></tr></thead><tbody>'
    for row in snapshot.get('annuals',[])[:5]:
        for key,label in LABELS.items():
            item=row.get(key)
            if not isinstance(item,dict):continue
            value=item.get('value')
            if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):continue
            acc=str(item.get('accession',''));cik=str(snapshot.get('cik',''))
            source=esc(acc or 'Accession unavailable')
            if re.fullmatch(r'\d{10}-\d{2}-\d{6}',acc) and re.fullmatch(r'\d{1,10}',cik):
                url=f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace("-", "")}/{acc}-index.html'
                source='<a href="'+url+'">'+source+'</a>'
            out+='<tr><td>'+esc(row.get('start',''))+' to '+esc(row.get('end',''))+'</td><td>'+esc(label)+': '+esc(str(value))+' '+esc(item.get('unit',''))+'</td><td>'+esc(item.get('calculation') or item.get('tag') or 'Definition unavailable')+'</td><td>'+esc(item.get('filed',''))+'<br>'+source+'</td></tr>'
    return out+'</tbody></table><p>Empty metrics are unavailable or withheld; no missing observations are interpolated. Filing links identify the source filing, not an independently verified physical page.</p>'
