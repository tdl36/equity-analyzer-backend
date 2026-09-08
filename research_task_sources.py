"""Resumable, bounded SEC source collection for a managed research command."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse,quote
import requests
from charlie_collector import Collector
from collection_refresh import RefreshManager
from sec_edgar import USER_AGENT


def get(url):
    u=urlparse(url)
    if u.scheme!='https' or u.hostname not in ('www.sec.gov','data.sec.gov') or u.username:raise ValueError('Only verified SEC endpoints are supported by this collector')
    with requests.get(url,headers={'User-Agent':USER_AGENT},timeout=25,stream=True,allow_redirects=False) as r:
        if r.status_code!=200:raise ValueError(f'SEC retrieval failed ({r.status_code}); this is not an empty filing search')
        data=bytearray()
        for block in r.iter_content(65536):
            data.extend(block)
            if len(data)>15_000_000:raise ValueError('SEC source exceeds 15 MB; no partial document saved')
        if not data:raise ValueError('SEC returned an empty document')
        return bytes(data)


def filing_sources(ticker,since,until,fetch=get):
    mapping=json.loads(fetch('https://www.sec.gov/files/company_tickers.json'))
    matches=[r for r in mapping.values() if r.get('ticker','').upper()==ticker]
    if len(matches)!=1:raise ValueError('Ticker could not be uniquely resolved to an SEC filer')
    cik=str(int(matches[0]['cik_str']));sub=json.loads(fetch(f'https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json'))
    recent=sub.get('filings',{}).get('recent',{});found=[]
    # Never silently interpret an unqueried historical archive as no filings.
    if any(f.get('filingFrom','')<=until and f.get('filingTo','')>=since for f in sub.get('filings',{}).get('files',[])):
        raise ValueError('Requested window overlaps historical SEC submissions; narrow to recent filings until archive retrieval is supported')
    for i,form in enumerate(recent.get('form',[])):
        day=recent['filingDate'][i]
        if form not in ('8-K','8-K/A') or not since<=day<=until:continue
        accession=recent['accessionNumber'][i];name=recent['primaryDocument'][i]
        if not re.fullmatch(r'\d{10}-\d{2}-\d{6}',accession) or not re.fullmatch(r'[A-Za-z0-9_.-]+',name):raise ValueError('Unexpected SEC filing identity')
        base=f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace("-","")}/'
        items=json.loads(fetch(base+'index.json')).get('directory',{}).get('item',[])
        selected=[name]+[x['name'] for x in items if re.search(r'ex(?:hibit)?[\-_]?99',x.get('name',''),re.I) and x['name'].lower().endswith(('.htm','.html','.pdf','.txt'))]
        for n in dict.fromkeys(selected):
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',n):raise ValueError('Unexpected SEC document filename')
            found.append({'url':base+quote(n),'name':f'SEC_{day}_{accession}_{n}','filingDate':day,'accession':accession})
    if len(found)>20:raise ValueError('More than 20 filing documents match; narrow the task date window')
    return found


def active(manager,rid,owner):
    row=manager.db.execute("SELECT * FROM refresh_requests WHERE id=? AND owner=? AND lease_until>? AND status='collecting'",(rid,owner,manager.clock())).fetchone()
    if not row:raise ValueError('Active collection lease required; no source write permitted')
    return row


def collect(manager,rid,owner,fetch=get):
    row=active(manager,rid,owner);cfg=json.loads(row['config']);p=cfg.get('researchCommand')
    if not p:raise ValueError('This request is not a research command')
    root=manager.c.catalysts/cfg['ticker']/cfg['topic']
    manager.db.execute('CREATE TABLE IF NOT EXISTS research_public_sources(request_id TEXT PRIMARY KEY,manifest TEXT)');manager.db.commit()
    previous=manager.db.execute('SELECT manifest FROM research_public_sources WHERE request_id=?',(rid,)).fetchone()
    if previous:
        saved=json.loads(previous['manifest'])
        if not all((root/d['filename']).is_file() and not (root/d['filename']).is_symlink() and hashlib.sha256((root/d['filename']).read_bytes()).hexdigest()==d['sha256'] for d in saved['documents']):
            raise ValueError('A previously collected public source changed or is missing; inspect the source register')
        if saved.get('complete'):return saved
        sources=saved['sources'];manifest=saved
    else:
        sources=filing_sources(cfg['ticker'],p['since'],p['until'],fetch)
        manifest={'documents':[],'sources':sources,'complete':False,'since':p['since'],'until':p['until'],'search':'SEC recent 8-K/8-K/A primary documents and filename-identified Exhibit 99 documents',
              'limitations':['Exhibit identification uses filenames; consult the filing index for other material attachments.','No matching 8-K does not mean no material company event.']}
        with manager.c.lock():
            active(manager,rid,owner)
            manager.db.execute('INSERT INTO research_public_sources VALUES(?,?)',(rid,json.dumps(manifest)))
    for source in sources:
        if any(d['url']==source['url'] for d in manifest['documents']):continue
        data=fetch(source['url']);digest=hashlib.sha256(data).hexdigest();name=source['name']
        if not Path(name).suffix.lower() in ('.html','.htm','.pdf','.txt'):name+='.html'
        with manager.c.lock():
            active(manager,rid,owner);target=root/name
            if target.is_symlink():raise ValueError('Refusing to replace a source symlink')
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Existing source differs; original retained')
            if not target.exists():
                temp=root/('.'+name+'.part');temp.write_bytes(data);temp.replace(target)
            manifest['documents'].append({**source,'filename':name,'sha256':digest})
            manager.db.execute('UPDATE research_public_sources SET manifest=? WHERE request_id=?',(json.dumps(manifest),rid))
    with manager.c.lock():
        active(manager,rid,owner)
        manifest['complete']=True
        manager.db.execute('UPDATE research_public_sources SET manifest=? WHERE request_id=?',(json.dumps(manifest),rid))
    return manifest


def verify_public_sources(manager,row,fetcher=None):
    manager.db.execute('CREATE TABLE IF NOT EXISTS research_public_sources(request_id TEXT PRIMARY KEY,manifest TEXT)');manager.db.commit()
    saved=manager.db.execute('SELECT manifest FROM research_public_sources WHERE request_id=?',(row['id'],)).fetchone()
    if not saved:raise ValueError('The SEC lookup has not been completed; run research_task_sources.py for this request')
    manifest=json.loads(saved['manifest'])
    if not manifest.get('complete'):raise ValueError('Public source retrieval is incomplete; resume before synthesis')
    documents=manifest['documents'];cfg=json.loads(row['config'])
    if fetcher is None:
        from charlie_local_agent import CHARLIE_API,_agent_headers
        def fetcher(ticker):
            r=requests.get(CHARLIE_API+'/api/agent/local-files/'+ticker,headers=_agent_headers(),timeout=25);r.raise_for_status();return r.json()
    files=fetcher(cfg['ticker']).get('files',[])
    for doc in documents:
        path=manager.c.catalysts/cfg['ticker']/cfg['topic']/doc['filename']
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=doc['sha256']:raise ValueError('Public source hash verification failed')
        if not any(f.get('filename')==doc['filename'] and f.get('folder')=='Catalysts/'+cfg['topic'] for f in files):raise ValueError('Public source is not yet visible in the production iCloud manifest')
    return len(documents)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--owner',required=True);parser.add_argument('--supplement',help='JSON file containing a verified FDA/registry source record');args=parser.parse_args()
    c=Collector()
    try:
        manager=RefreshManager(c)
        if args.supplement:
            from trusted_research_sources import register
            result=register(manager,args.request,args.owner,json.loads(Path(args.supplement).read_text()))
        else:result=collect(manager,args.request,args.owner)
        print(json.dumps(result,indent=2))
    finally:c.db.close()
