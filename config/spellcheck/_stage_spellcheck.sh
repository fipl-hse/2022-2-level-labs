set -ex

echo -e '\n'

echo "Spellchek running ..."

source venv/bin/activate

echo "$PYTHONPATH"
echo "$PATH"

python -m pyspelling -c config/spellcheck/.spellcheck.yaml -v
