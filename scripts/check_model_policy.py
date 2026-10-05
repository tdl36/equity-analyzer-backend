#!/usr/bin/env python3
"""Offline release gate. Never calls a model or opens the application database."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', help='Git revision containing the previously approved registry')
    parser.add_argument('--evidence', help='JSON evidence keyed by changed role')
    args = parser.parse_args()
    errors = model_registry.validate(model_registry.REGISTRY)
    if args.baseline:
        raw = subprocess.check_output(['git','show', f'{args.baseline}:model_registry.json'], text=True)
        before = json.loads(raw)
        evidence = json.loads(Path(args.evidence).read_text()) if args.evidence else {}
        errors += model_registry.migration_errors(before, model_registry.REGISTRY, evidence)
    for error in sorted(set(errors)): print(error)
    if errors: return 1
    print('Model policy passed. Offline checks do not establish real-source quality or future token cost.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
