#!/bin/bash
export CHARLIE_API_KEY=ejpF14PQz2Wt8NxyHwoMVN_hlxfuPMNU3GHwjA5-2_Q
unset PYTHONPATH
unset PYTHONHOME
eval "$(grep 'export ANTHROPIC_API_KEY=' ~/.zshrc 2>/dev/null | head -1)"
eval "$(grep 'export TELEGRAM_' ~/.zshrc 2>/dev/null)"
PYTHON=/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python
exec $PYTHON /Users/tonydlee/Projects/equity-analyzer-backend/charlie_local_agent.py
