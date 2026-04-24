#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/Scripts/activate
python main.py
