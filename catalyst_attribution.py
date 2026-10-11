"""Conservative headline attribution hints, not primary-source verification."""
import re
_SUFFIXES={'inc','incorporated','corp','corporation','ltd','limited','plc','llc','co','company','group','holdings'}
def normalized(value):return re.sub(r'[^\w]+',' ',str(value or '').casefold()).strip()
def attribution(ticker,title,company_name=None):
    headline=normalized(title);name=normalized(company_name).split()
    while name and name[-1] in _SUFFIXES:name.pop()
    if name and name[-1]=='and':name.pop()
    if name and name[0]=='the':name=name[1:]
    name=' '.join(name)
    # Short symbols can be ordinary headline words. Require exchange notation or
    # a usable saved company name instead of treating one/two letters as identity.
    symbol=bool(re.match(r'^(?:(?:NYSE|NASDAQ)\s*:\s*'+re.escape(ticker)+r'\b|\('+re.escape(ticker)+r'\))',title.strip(),re.I))
    if len(ticker)>=3:symbol=symbol or bool(re.match(r'^'+re.escape(ticker)+r'\b',title.strip()))
    named=bool(len(name)>=3 and (headline==name or headline.startswith(name+' ')))
    relationship=bool(re.search(r'\b(?:rival|competitor|customer|supplier|partner|shareholder|investor)\b',headline[:100]))
    matched=(symbol or named) and not relationship
    return {'matched':matched,'basis':'related_party_headline' if relationship else 'ticker_prefix' if symbol else 'saved_company_name_prefix' if named else 'unverified'}
