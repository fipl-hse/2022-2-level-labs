#!/bin/bash

set -ex

echo -e '\n'
echo 'Running lint check...'

source venv/bin/activate

export PYTHONPATH="$(pwd):${PYTHONPATH}"

python -m pylint --rcfile config/stage_1_style_tests/.pylintrc config seminars

FAILED=0
LABS=$(cat config/labs.txt)

for LAB_NAME in $LABS; do
	echo "Running lint for lab ${LAB_NAME}"
  TARGET_SCORE=$(bash config/get_mark.sh ${LAB_NAME})

	lint_output="$(python -m pylint --rcfile config/stage_1_style_tests/.pylintrc ${LAB_NAME})"

  python config/stage_1_style_tests/lint_level.py \
          --lint-output "${lint_output}" \
          --target-score "${TARGET_SCORE}"

  if [[ $? -ne 0 ]]; then
    echo "Lint check failed for lab ${LAB_NAME}."
    FAILED=1
  else
    echo "Lint check passed for lab ${LAB_NAME}."
  fi
done

if [[ ${FAILED} -eq 1 ]]; then
	echo "Lint check failed."
	exit ${FAILED}
fi

echo "Lint check passed."
