#!/usr/bin/env python3
"""Install Charlie's loopback collection monitor as a per-user macOS service."""
import os
from pathlib import Path
import plistlib
import subprocess

root = Path(__file__).resolve().parents[1]
python = root / '.venv/bin/python'
if not python.exists():
    raise SystemExit('Create the project .venv before installing the monitor.')
state = Path.home() / 'Library/Application Support/Charlie/AlphaSense'
state.mkdir(parents=True, exist_ok=True, mode=0o700)
label = 'com.charlie.collector-monitor'
plist = Path.home() / 'Library/LaunchAgents' / (label + '.plist')
plist.parent.mkdir(parents=True, exist_ok=True)
settings = {
    'Label': label,
    'ProgramArguments': [str(python), str(root / 'collector_status.py')],
    'WorkingDirectory': str(root),
    'RunAtLoad': True,
    'KeepAlive': True,
    'ThrottleInterval': 15,
    'StandardOutPath': str(state / 'monitor.log'),
    'StandardErrorPath': str(state / 'monitor-error.log'),
}
# This label belongs only to this monitor; never restart the production file agent.
service = f'gui/{os.getuid()}/{label}'
subprocess.run(['launchctl', 'bootout', service], capture_output=True)
plist.write_bytes(plistlib.dumps(settings))
subprocess.run(['launchctl', 'bootstrap', f'gui/{os.getuid()}', str(plist)], check=True)
print('Read-only collection monitor installed at http://127.0.0.1:8766/')
