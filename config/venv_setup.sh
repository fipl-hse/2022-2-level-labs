#!/bin/bash

set -ex

python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv venv

venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pip install -r requirements_qa.txt
