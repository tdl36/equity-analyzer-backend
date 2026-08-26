"""Runs the frontend renderer's escaping tests as part of the normal suite.

The escaping contract is frontend JS, but it is a security property, so it
should not live behind a separate command nobody remembers to run.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / 'render_markdown_test.mjs'


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_markdown_renderer_escaping_contract():
    result = subprocess.run(['node', str(SCRIPT)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
