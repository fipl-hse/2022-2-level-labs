set -ex

echo -e '\n'

echo "Spellchek running ..."

source venv/bin/activate

echo "PYTHONPATH: $PYTHONPATH"
echo "PATH $PATH"
echo "WHICH PYTHON: "
which python

python -m pyspelling -c config/spellcheck/.spellcheck.yaml -v
