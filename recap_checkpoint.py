"""Atomic partial recap checkpoints. Completed drafts use the delivery outbox."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

DEFAULT_ROOT=Path.home()/'Library/Application Support/Charlie/RecapCheckpoints'

class Checkpoint:
    def __init__(self, inputs, root=DEFAULT_ROOT):
        self.key=hashlib.sha256(json.dumps(inputs,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        self.root=Path(root);self.path=self.root/(self.key+'.json')
    def load(self, total):
        try:
            if self.path.stat().st_size>2000000:return None
            value=json.loads(self.path.read_text())
            if value.get('key')!=self.key or type(value.get('completed'))!=int or not 1<=value['completed']<=total or not isinstance(value.get('markdown'),str) or not value['markdown'].strip():return None
            return value
        except (OSError,ValueError,TypeError,AttributeError):return None
    def save(self,completed,markdown):
        if not markdown.strip():raise ValueError('Cannot checkpoint an empty recap')
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        name=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',dir=self.root,delete=False) as stream:
                name=stream.name;json.dump({'key':self.key,'completed':completed,'markdown':markdown},stream);stream.flush();os.fsync(stream.fileno())
            os.replace(name,self.path)
        finally:
            if name and os.path.exists(name):os.unlink(name)
    def finish(self):self.path.unlink(missing_ok=True)
