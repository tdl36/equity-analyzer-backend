#!/usr/bin/env python3
"""Exercise actual PostgreSQL ownership in a disposable LOCAL schema, no model calls."""
import sys,json,uuid,threading
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql
from manual_meeting_recovery import execute

schema='charlie_recovery_qa_'+uuid.uuid4().hex
# Deliberately ignores application URLs and credentials: Unix socket, local postgres only.
admin=psycopg2.connect(dbname='postgres',host='/tmp');admin.autocommit=True
@contextmanager
def db(commit=False):
    conn=psycopg2.connect(dbname='postgres',host='/tmp')
    try:
        cur=conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
        yield conn,cur
        if commit:conn.commit()
    finally:conn.close()
try:
    with admin.cursor() as cur:
        cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        cur.execute(sql.SQL("CREATE TABLE {}.mp_jobs(id text primary key,stage text,status text,input jsonb,result jsonb,error text,updated_at timestamp default now())").format(sql.Identifier(schema)))
    with db(True) as (_,cur):
        cur.execute("INSERT INTO mp_jobs VALUES('fixture','pipeline','running','{}','{\"analyses\":[{\"done\":true},null]}',NULL,NOW())")
    entered=threading.Event();release=threading.Event();errors=[];old_owner=[]
    def interrupted(inp,checkpoint):
        old_owner.append(inp['workerToken']);entered.set();release.wait(10)
        raise RuntimeError('Simulated process interruption after checkpoint')
    def first():
        try:execute(db,'fixture',interrupted)
        except RuntimeError:pass
        except Exception as e:errors.append(str(e))
    worker=threading.Thread(target=first);worker.start();assert entered.wait(10)
    assert execute(db,'fixture',lambda *_:errors.append('duplicate execution')) is False
    release.set();worker.join(10);assert not worker.is_alive();assert not errors,errors
    def resumed(inp,checkpoint):
        assert checkpoint['analyses']==[{'done':True},None]
        assert inp['recoveryAttempts']==1 and inp['workerToken']!=old_owner[0]
        with db(True) as (_,cur):
            cur.execute("UPDATE mp_jobs SET status='failed' WHERE id='fixture' AND input->>'workerToken'=%s",(old_owner[0],))
            assert cur.rowcount==0,'Stale owner changed recovered job'
            cur.execute("UPDATE mp_jobs SET status='done' WHERE id='fixture'")
    assert execute(db,'fixture',resumed,True)
    assert execute(db,'fixture',lambda *_:errors.append('completed replay'),True) is False
    assert not errors,errors
    import base64,hashlib
    from manual_meeting_sources import freeze
    from meeting_source_support import excerpts
    with db(True) as (_,cur):
        cur.execute('CREATE TABLE mp_documents(id int,meeting_id int,filename text,file_data text,extracted_text text,doc_type text)')
        text='Original source evidence describing margin growth and financial performance.'
        cur.execute('INSERT INTO mp_documents VALUES(1,1,%s,%s,%s,%s)',('source.txt',base64.b64encode(text.encode()).decode(),text,'report'))
    frozen=freeze(db,1,[{'id':1}])
    assert frozen[0]['sha256']==hashlib.sha256(text.encode()).hexdigest()
    evidence=excerpts(db,frozen)
    assert evidence['sources'][0]['pages'][0]['text']==text
    # Exercise the actual save function in the same isolated schema.
    import ast
    with db(True) as (_,cur):
        cur.execute("CREATE TABLE mp_meetings(id int primary key,company_id int,status text,updated_at timestamp)")
        cur.execute("CREATE TABLE mp_question_sets(id serial primary key,meeting_id int,version int,status text,topics_json text,synthesis_json text,generation_model text,generation_tokens int)")
        cur.execute("CREATE TABLE mp_past_questions(id serial primary key,company_id int,meeting_id int,question text,topic text,status text)")
        cur.execute("INSERT INTO mp_meetings VALUES(1,1,'draft',NOW())")
        cur.execute("UPDATE mp_jobs SET status='running',input=%s::jsonb WHERE id='fixture'",(json.dumps({'workerToken':'save-owner'}),))
    tree=ast.parse((Path(__file__).resolve().parents[1]/'app_v3.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mp_save_results_inline')
    scope={'get_db':db,'json':json}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'app_v3.py','exec'),scope)
    save=scope['_mp_save_results_inline']
    args=(1,[{'topic':'QA','questions':[{'question':'Fixture question?'}]}],{},0,'fixture')
    first_save=save(*args,job_id='fixture',owner='save-owner')
    assert save(*args,job_id='fixture',owner='save-owner')==first_save
    with db() as (_,cur):
        cur.execute('SELECT count(*) AS n FROM mp_question_sets');assert cur.fetchone()['n']==1
        cur.execute('SELECT status FROM mp_past_questions');assert cur.fetchone()['status']=='planned'
    print(json.dumps({'passed':['active worker excluded','partial checkpoint retained','owner rotated after interruption','stale owner write rejected','completed job not replayed','atomic save receipt prevents duplicate versions','generated questions marked planned','server source freezing and sequential extraction'],'scope':'Disposable local PostgreSQL schema; no live research or model calls.'},indent=2))
finally:
    with admin.cursor() as cur:cur.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
    admin.close()
