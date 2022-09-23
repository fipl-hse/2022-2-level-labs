#!/bin/bash

set -ex

source venv/bin/activate

which python

export PYTHONPATH="$(pwd):${PYTHONPATH}"

python config/collect_coverage/coverage_analyzer.py
