#!/bin/bash

set -ex

source venv/bin/activate

echo "Running start.py checks..."

source venv/bin/activate

export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "PYTHONPATH: $PYTHONPATH"
echo "PATH $PATH"
echo "WHICH PYTHON: "
which python

echo "LS -LA venv/bin"
ls -la venv/bin

LABS=$(cat config/labs.txt)
WAS_FAILED=0

for LAB_NAME in $LABS; do
	echo "Running start.py checks for lab ${LAB_NAME}"

	if ! python ${LAB_NAME}/start.py;  then
    	WAS_FAILED=1
	fi

	if [[ $WAS_FAILED -eq 1 ]]; then
    echo "start.py fails while running"
    echo "Check for start.py file for lab ${LAB_NAME} failed."
    exit 1
  fi

  echo "Check calling lab ${LAB_NAME} passed"

  START_PY_FILE=$(cat ${LAB_NAME}/start.py)
  python config/check_start_content.py --start_py_content "$START_PY_FILE"
done
