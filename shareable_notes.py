"""Shared reader-facing contract for Summary, Lab and webhook inputs."""
import json
from pathlib import Path
RULES = '\n\n' + json.loads(Path(__file__).with_name('shareable-note-policy.json').read_text())['instruction']
