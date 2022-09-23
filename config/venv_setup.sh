#!/bin/bash

set -ex

which python

if [ -f "venv/bin/python" ]; then
  echo "VENV and Python already exist."
  venv/bin/python -m pip install -r requirements.txt
  venv/bin/python -m pip install -r requirements_qa.txt
  exit 0
fi

if [ -d "venv" ]; then
  echo "VENV exists but not Python exec. Setup a new venv."
  rm -rf venv
fi

python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv venv

sleep 10

source venv/bin/activate

which python

python -m pip install -r requirements.txt
python -m pip install -r requirements_qa.txt
