#!/bin/bash

set -ex

source venv/bin/activate

export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "PYTHONPATH: $PYTHONPATH"
echo "PATH $PATH"
echo "WHICH PYTHON: "
which python

python config/collect_coverage/coverage_analyzer.py
