#!/bin/bash

set -ex

export PYTHONPATH="$(pwd):${PYTHONPATH}"

venv/bin/python config/collect_coverage/coverage_analyzer.py
