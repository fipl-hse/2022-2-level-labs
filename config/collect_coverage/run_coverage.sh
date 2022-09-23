#!/bin/bash

set -ex

source venv/bin/activate

export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "$PYTHONPATH"
echo "$PATH"

python config/collect_coverage/coverage_analyzer.py
