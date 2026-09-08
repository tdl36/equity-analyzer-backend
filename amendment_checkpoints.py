"""Durable proposal stage outputs, keyed to complete prompts and original inputs."""
import hashlib
import json


def identity(prompt, hashes, model=None):
    return hashlib.sha256(json.dumps({'version':1,'prompt':prompt,'sourceHashes':hashes,'model':model},sort_keys=True).encode()).hexdigest()


class Checkpoints:
    def __init__(self,get_db,job_id,owner=None):self.get_db=get_db;self.job_id=job_id;self.owner=owner
    def load(self,key,stage):
        with self.get_db() as (_,cur):
            cur.execute("SELECT status,result,input->>'workerToken' AS owner FROM mp_jobs WHERE id=%s AND stage='evidence_amendment'",(self.job_id,))
            row=cur.fetchone()
        if not row or row['status']!='running' or (self.owner and row.get('owner')!=self.owner):raise ValueError('Proposal was stopped; no additional model work permitted')
        result=row.get('result') or {}
        if isinstance(result,str):result=json.loads(result)
        saved=result.get('checkpoint',{})
        if saved and saved.get('key')!=key:raise ValueError('Proposal inputs changed; start a fresh comparison instead of resuming')
        return saved.get(stage)
    def save(self,key,stage,value):
        if stage not in ('draft','review'):raise ValueError('Unknown proposal checkpoint stage')
        encoded=json.dumps(value)
        if len(encoded)>1_000_000:raise ValueError('Proposal stage output exceeds checkpoint limit')
        with self.get_db(commit=True) as (_,cur):
            cur.execute("SELECT status,result,input->>'workerToken' AS owner FROM mp_jobs WHERE id=%s AND stage='evidence_amendment' FOR UPDATE",(self.job_id,))
            row=cur.fetchone()
            if not row or row['status']!='running' or (self.owner and row.get('owner')!=self.owner):raise ValueError('Proposal was stopped; late stage output was not saved')
            result=row.get('result') or {}
            if isinstance(result,str):result=json.loads(result)
            saved=result.get('checkpoint',{})
            if saved and saved.get('key')!=key:raise ValueError('Proposal checkpoint inputs differ')
            if stage in saved and saved[stage]!=value:raise ValueError('A different stage output is already saved')
            result['checkpoint']={**saved,'key':key,stage:value}
            cur.execute("UPDATE mp_jobs SET result=%s::jsonb,updated_at=NOW() WHERE id=%s",(json.dumps(result),self.job_id))
