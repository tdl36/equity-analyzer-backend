"""Isolated, bounded PDF passage extraction for meeting preparation."""
import base64,json,subprocess,sys,threading
from pathlib import Path
_SLOTS=threading.BoundedSemaphore(2)


def extract_original(encoded,timeout=40):
    if len(encoded)>80_000_000:raise ValueError('PDF exceeds the passage extraction size limit.')
    with _SLOTS:
        try:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--extract'],
                input=base64.b64decode(encoded,validate=True),stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=timeout,check=True)
            return json.loads(result.stdout)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,ValueError) as exc:
            raise ValueError('Original PDF passage extraction exceeded its time/resource limit or failed.') from exc


def main():
    if sys.platform.startswith('linux'):
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(384*1024*1024,384*1024*1024))
    from pdf_text import extract
    value=extract(sys.stdin.buffer.read(),strict=False)
    # Bound the returned text even for exceptionally long originals.
    remaining=500000;pages=[]
    for page in value['pages']:
        if remaining<=0:break
        text=page['text'][:remaining];remaining-=len(text)
        pages.append({'page':page['page'],'text':text})
    if remaining<=0:value['limitations'].append('Original text extraction capped at 500,000 characters.')
    print(json.dumps({'pages':pages,'limitations':value['limitations']}))

if __name__=='__main__':main()
