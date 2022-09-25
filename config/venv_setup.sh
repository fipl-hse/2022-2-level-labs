#!/bin/bash

set -ex

which python

python -m pip install --upgrade pip
python -m pip install virtualenv==20.16.4
python -m virtualenv venv --always-copy

source venv/bin/activate

which python

python -m pip install -r requirements.txt
python -m pip install -r requirements_qa.txt
