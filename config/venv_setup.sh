#!/bin/bash

set -ex

which python

python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv venv

source venv/bin/activate

echo "PYTHONPATH: $PYTHONPATH"
echo "PATH $PATH"
echo "WHICH PYTHON: "
which python

echo "LS -LA venv/bin"
ls -la venv/bin

python -m pip install -r requirements.txt
python -m pip install -r requirements_qa.txt
