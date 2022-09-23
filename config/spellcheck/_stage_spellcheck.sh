set -ex

echo -e '\n'

echo "Spellchek running ..."

source venv/bin/activate

echo "PYTHONPATH: $PYTHONPATH"
echo "PATH $PATH"
echo "WHICH PYTHON: "
which python

echo "LS -LA venv/bin"
ls -la venv/bin

python -m pyspelling -c config/spellcheck/.spellcheck.yaml -v
