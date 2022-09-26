#!/bin/bash

set -ex

source venv/bin/activate

export PYTHONPATH="$(pwd):${PYTHONPATH}"

IS_ADMIN=$(python config/is_admin.py --pr_name "$1")
if [ "$REPOSITORY_TYPE" == "public" ] && [ "$IS_ADMIN" == 'YES' ] ; then
  echo '[skip-lab] option was enabled, skipping check...'
  exit 0
fi

python config/collect_coverage/coverage_analyzer.py
