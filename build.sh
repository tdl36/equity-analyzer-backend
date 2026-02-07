#!/usr/bin/env bash
# Render build script

set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
