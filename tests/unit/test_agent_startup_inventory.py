"""A failed inventory must never trigger original reads or bulk reuploads."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock


class StartupInventoryTests(TestCase):
    def run_import(self, name, status=200, payload=None, error=None):
        tree=ast.parse(Path('charlie_local_agent.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        request=Mock();request.get.return_value=Mock(status_code=status)
        request.get.return_value.json.return_value=payload
        if error:request.get.side_effect=error
        read=Mock(side_effect=AssertionError('Original read on failed inventory'))
        with TemporaryDirectory() as root:
            folder=Path(root)/'TEST';folder.mkdir()
            (folder/'source.pdf').write_bytes(b'synthetic')
            (folder/'TEST_Note_1.md').write_text('synthetic')
            scope={'STOCKS_DIR':Path(root),'requests':request,'CHARLIE_API':'https://test.invalid',
                   '_agent_headers':lambda:{},'log':Mock(),'MAX_SYNC_BYTES':1000,'_read_icloud_bytes':read}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'charlie_local_agent.py','exec'),scope)
            scope[name]()
        request.post.assert_not_called();read.assert_not_called()
        return scope['log']

    def test_failed_or_malformed_document_inventory_is_not_empty(self):
        for status,payload,error in [(502,None,None),(401,{},None),(404,{},None),(200,{},None),
             (200,{'documents':None},None),(200,{'documents':[{}]},None),(200,[],None),
             (200,None,TimeoutError())]:
            with self.subTest(status=status,payload=payload,error=error):
                self.run_import('import_existing_documents',status,payload,error).warning.assert_called()

    def test_existing_document_is_not_read(self):
        self.run_import('import_existing_documents',200,{'documents':[{'filename':'source.pdf'}]})

    def test_notes_only_known_absence_allows_scan(self):
        for status,payload,error in [(502,None,None),(401,{},None),(404,{},None),(200,None,TimeoutError())]:
            with self.subTest(status=status,payload=payload):
                self.run_import('import_existing_notes',status,payload,error).warning.assert_called()
        self.run_import('import_existing_notes',200,{'id':'existing'})
