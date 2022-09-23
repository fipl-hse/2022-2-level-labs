set -ex

echo -e '\n'

echo "Pull Request Name is $1"

source venv/bin/activate

python config/stage_1_style_tests/pr_name_check.py --pr-name="$1" --pr-author="$2"
