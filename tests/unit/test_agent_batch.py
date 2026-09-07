import importlib.util
from pathlib import Path
import threading
import time
import unittest

spec=importlib.util.spec_from_file_location('agent_batch',Path(__file__).parents[2]/'agent_batch.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class BatchTests(unittest.TestCase):
    def test_validation_and_deduplication(self):
        self.assertEqual(mod.validate_batch({'tickers':['de',' DE ','BRK.B'],'concurrency':2}),(['DE','BRK.B'],2))
        for data in [{'tickers':[]},{'tickers':'DE'},{'tickers':['DE!']},{'tickers':['DE'],'concurrency':True},{'tickers':['DE'],'concurrency':4},{'tickers':[str(i) for i in range(13)]}]:
            with self.assertRaises(ValueError):mod.validate_batch(data)
    def test_parallel_execution_is_bounded_and_failure_isolated(self):
        lock=threading.Lock();active=0;peak=0;done=[];failures=[]
        def run(job):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            try:
                time.sleep(.03)
                if job==2:raise RuntimeError('test failure')
                done.append(job)
            finally:
                with lock:active-=1
        mod.run_batch_jobs(range(8),run,3,lambda job,exc:failures.append(job))
        self.assertGreater(peak,1);self.assertLessEqual(peak,3)
        self.assertEqual(sorted(done),[0,1,3,4,5,6,7]);self.assertEqual(failures,[2])
    def test_multiple_batches_share_the_process_cap(self):
        lock=threading.Lock();active=0;peak=0
        def run(job):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            time.sleep(.02)
            with lock:active-=1
        batches=[threading.Thread(target=mod.run_batch_jobs,args=(range(6),run,3)) for _ in range(2)]
        for t in batches:t.start()
        for t in batches:t.join()
        self.assertLessEqual(peak,3)
    def test_default_is_sequential(self):
        result=[];mod.run_batch_jobs(range(4),result.append)
        self.assertEqual(result,list(range(4)))

if __name__=='__main__':unittest.main()
