#!/usr/bin/env bash
# Render build script

set -o errexit

# Install system dependencies (ffmpeg for pydub audio processing)
apt-get update && apt-get install -y --no-install-recommends ffmpeg

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
