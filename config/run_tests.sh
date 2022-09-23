#!/bin/bash

set -ex

echo "Running tests..."

source venv/bin/activate

echo "PYTHONPATH: $PYTHONPATH"
echo "PATH $PATH"
echo "WHICH PYTHON: "
which python

echo "LS -LA venv/bin"
ls -la venv/bin

FAILED=0
LABS=$(cat config/labs.txt)

echo "Current scope: $LABS"

for LAB_NAME in $LABS; do
  echo "Running tests for lab ${LAB_NAME}"

	TARGET_SCORE=$(bash config/get_mark.sh ${LAB_NAME})

	if [[ ${TARGET_SCORE} == 0 ]]; then
    echo "Skipping stage"
    continue
  fi

	if ! python -m pytest -m "mark${TARGET_SCORE} and ${LAB_NAME}"; then
    	FAILED=1
	fi
done

if [[ ${FAILED} -eq 1 ]]; then
	echo "Tests failed."
	exit 1
fi

echo "Tests passed."
