set -ex

echo -e '\n'

echo "Spellchek running ..."

venv/bin/python -m pyspelling -c config/spellcheck/.spellcheck.yaml -v
