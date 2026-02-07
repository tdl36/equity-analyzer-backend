#!/usr/bin/env bash
# Render build script - installs system dependencies and Python packages

set -o errexit

# Install ffmpeg (required by pydub for audio compression)
apt-get update && apt-get install -y ffmpeg

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
