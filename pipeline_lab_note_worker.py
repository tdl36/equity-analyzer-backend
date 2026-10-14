"""One bounded Lab note worker iteration; not scheduled in production.

All model calls remain injected. Background lease renewal never invokes a model.
"""
import threading
from pipeline_lab_note_jobs import claim, renew, settle
from pipeline_lab_note_run import execute


def process_one(get_db, load_documents, invoke, planner=None, executor=execute):
    with get_db(commit=True) as (_,cur):
        owned=claim(cur)
    if owned is None:return None
    note_id=owned['noteId'];lease_id=owned['leaseId']
    stop=threading.Event();lost=threading.Event()

    def check():
        if lost.is_set():raise ValueError('Worker ownership was lost.')
        try:
            with get_db(commit=True) as (_,cur):renew(cur,note_id,lease_id)
        except Exception:
            lost.set()
            raise ValueError('Worker ownership could not be verified.') from None

    def keep_alive():
        while not stop.wait(30):
            try:check()
            except Exception:return

    def originals(plan):
        check()
        return load_documents(plan)

    def provider(**kwargs):
        check()  # Ownership must still be current before every new paid call.
        return invoke(**kwargs)

    thread=threading.Thread(target=keep_alive,name='lab-note-lease',daemon=True)
    thread.start()
    try:
        try:
            check()
            result=executor(get_db,note_id,originals,provider,planner)
        except ValueError:
            result={'status':'attention','reason':'worker_interrupted' if lost.is_set() else 'source_check_failed'}
        except Exception:
            # A journaled call remains ambiguous; never expose raw exceptions.
            result={'status':'attention','reason':'worker_interrupted'}
        try:
            check()
            with get_db(commit=True) as (_,cur):
                return settle(cur,note_id,lease_id,result)
        except Exception:
            # Do not claim saved settlement if the lease/commit cannot be verified.
            # The queue reaper and explicit recovery reconcile durable evidence.
            return {'noteId':note_id,'status':'attention','reason':'worker_interrupted'}
    finally:
        stop.set()
        thread.join(timeout=5)
