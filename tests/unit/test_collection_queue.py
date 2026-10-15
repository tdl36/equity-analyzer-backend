import sqlite3
import unittest
from collection_queue import queue_diagnostics

class QueueDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(':memory:');self.db.row_factory=sqlite3.Row;self.addCleanup(self.db.close)
        self.db.executescript('''CREATE TABLE refresh_requests(id TEXT,ticker TEXT,status TEXT,created REAL,lease_until REAL,config TEXT);
            CREATE TABLE refresh_policies(ticker TEXT,config TEXT);
            CREATE TABLE refresh_claim_order(request_id TEXT,turn INTEGER);''')
    def row(self, ident, status='queued', lease=None, enabled=1, manual=0, turn=0):
        self.db.execute('INSERT INTO refresh_requests VALUES(?,?,?,?,?,?)',(ident,ident,status,1,lease,'{"manual":'+str(manual)+'}'))
        if enabled is not None:self.db.execute('INSERT INTO refresh_policies VALUES(?,?)',(ident,'{"enabled":'+str(enabled)+'}'))
        self.db.execute('INSERT INTO refresh_claim_order VALUES(?,?)',(ident,turn))
    def test_read_only_fair_order_and_global_slot(self):
        self.row('old',turn=4);self.row('new');self.row('active','collecting',200,turn=5)
        changes=self.db.total_changes;result=queue_diagnostics(self.db,100)
        self.assertEqual(self.db.total_changes,changes)
        self.assertEqual(result['new']['defaultPosition'],1);self.assertEqual(result['old']['defaultPosition'],2)
        self.assertEqual(result['new']['state'],'waiting_for_slot');self.assertEqual(result['active']['state'],'leased')
        self.assertEqual(result['new']['active'],[dict(requestId='active',ticker='active',leaseUntil=200)])
        self.assertNotIn('owner',str(result));self.assertNotIn('config',str(result))
    def test_pause_manual_missing_policy_attention_and_queued_lease(self):
        self.row('paused',enabled=0);self.row('manual',enabled=0,manual=1);self.row('missing',enabled=None,manual=1)
        self.row('attention','attention');self.row('auth','needs_auth');self.row('lease',lease=200)
        r=queue_diagnostics(self.db,100)
        self.assertEqual(r['paused']['state'],'policy_paused');self.assertEqual(r['manual']['state'],'eligible')
        self.assertEqual(r['missing']['state'],'missing_policy');self.assertEqual(r['attention']['state'],'needs_attention')
        self.assertEqual(r['auth']['state'],'needs_attention');self.assertEqual(r['lease']['state'],'lease_wait')
        self.assertEqual(r['manual']['eligibleCount'],1)
    def test_expired_lease_enters_default_order_without_renewing(self):
        self.row('expired','verifying',100);self.row('done','complete');self.row('cancel','cancelled')
        r=queue_diagnostics(self.db,100)
        self.assertEqual(list(r),['expired']);self.assertEqual(r['expired']['state'],'eligible')
        self.assertEqual(self.db.execute('SELECT lease_until FROM refresh_requests WHERE id="expired"').fetchone()[0],100)
    def test_multiple_active_leases_are_not_hidden_or_repaired(self):
        for i in range(5):self.row(str(i),'collecting',200)
        self.row('waiting');r=queue_diagnostics(self.db,100)['waiting']
        self.assertEqual(r['activeCount'],5);self.assertEqual(len(r['active']),3)
        self.assertEqual(r['state'],'waiting_for_slot')
