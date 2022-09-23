#!/bin/bash

set -ex

echo -e '\n'
echo 'Running stubgen check...'

source venv/bin/activate

echo "$PYTHONPATH"
echo "$PATH"

export PYTHONPATH="$(pwd):${PYTHONPATH}"

FAILED=0
LABS=$(cat config/labs.txt)

for LAB_NAME in $LABS; do
	echo "Running stubgen for lab ${LAB_NAME}"

  python ./config/generate_stubs/run_generator.py \
          --source_code_path ./${LAB_NAME}/main.py \
          --target_code_path ./build/stubs/main.py

  if [[ $? -ne 0 ]]; then
    echo "Stubgen check failed for lab ${LAB_NAME}."
    FAILED=1
  else
    echo "Stubgen check passed for lab ${LAB_NAME}."
  fi
done

if [[ ${FAILED} -eq 1 ]]; then
	echo "Stubgen check failed."
	exit ${FAILED}
fi

echo "Stubgen check passed."
