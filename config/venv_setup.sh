#!/bin/bash

set -ex

which python

if [ -d "venv" ]; then
  echo "VENV already exists."
  venv/bin/python -m pip install -r requirements.txt
  venv/bin/python -m pip install -r requirements_qa.txt
  exit 0
fi

python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv venv

sleep 10

which python

venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pip install -r requirements_qa.txt
