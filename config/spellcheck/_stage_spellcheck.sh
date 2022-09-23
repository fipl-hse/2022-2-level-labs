set -ex

echo -e '\n'

echo "Spellchek running ..."

source venv/bin/activate

which python

python -m pyspelling -c config/spellcheck/.spellcheck.yaml -v
