import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import audio_issue_notifications as issues
from tests.unit.test_audio_intake_receipts import load, AGENT

class IssueNotificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'intake.db'

    def test_confirmed_delivery_deduplicates_across_repeated_issues(self):
        send=Mock(return_value=True)
        for _ in range(3):
            issues.enqueue('**UNH_[CFO].m4a','Processing failed.','job',self.path)
            issues.deliver(send,self.path,now=1)
        self.assertEqual(send.call_count,1)
        self.assertIn('**UNH_[CFO].m4a',send.call_args.args[0])
        issues.enqueue('**UNH_[CFO].m4a','Processing failed.','new-job',self.path)
        issues.deliver(send,self.path,now=2)
        self.assertEqual(send.call_count,2)

    def test_transport_failure_or_unacknowledged_send_retries_after_delay(self):
        for sender in [Mock(side_effect=TimeoutError),Mock(return_value=False)]:
            with TemporaryDirectory() as d:
                path=Path(d)/'db'
                issues.enqueue('audio.m4a','Status unknown.','job',path)
                issues.deliver(sender,path,now=0)
                successful=Mock(return_value=True)
                issues.deliver(successful,path,now=299)
                successful.assert_not_called()
                issues.deliver(successful,path,now=300)
                successful.assert_called_once()

    def test_telegram_uses_plain_text_and_checks_response(self):
        http=Mock();http.post.return_value.json.return_value={'ok':False}
        delivery=Mock()
        scope={'TELEGRAM_BOT_TOKEN':'fixture-token','TELEGRAM_CHAT_ID':'fixture-chat',
               'requests':http,'audio_issue_notifications':delivery,'log':Mock()}
        fn=load(AGENT,'_deliver_audio_issues',scope);fn()
        sender=delivery.deliver.call_args.args[0]
        self.assertFalse(sender('**UNH_[CFO].m4a'))
        self.assertNotIn('parse_mode',http.post.call_args.kwargs['json'])
        http.post.return_value.json.return_value={'ok':True}
        self.assertTrue(sender('Failure'))
        http.post.return_value.raise_for_status.assert_called()

    def test_cloud_failure_does_not_prevent_telegram_queue(self):
        receipt={'id':'receipt','job_id':'job','status':'attention'}
        api=Mock();api.update.return_value=receipt;api.reserve_alert.return_value=True
        http=Mock();http.post.side_effect=TimeoutError
        queue=Mock()
        fn=load(AGENT,'_hold_audio_intake',{'audio_intake_receipts':api,
                '_queue_audio_issue':queue,'requests':http,'CHARLIE_API':'https://invalid.test',
                '_agent_headers':lambda:{},'log':Mock()})
        fn(receipt,'UNH.m4a','Processing failed.')
        queue.assert_called_once_with('UNH.m4a','Processing failed.','job')
        api.update.return_value={**receipt,'status':'complete'}
        queue.reset_mock();fn(receipt,'UNH.m4a','Processing failed.');queue.assert_not_called()
