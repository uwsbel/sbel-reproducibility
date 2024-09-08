#!/usr/bin/env bash

#setup python virtual with python3 
if python3 -m venv ~/.venv/cpd; then
    #install deps
    . ~/.venv/cpd/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Error creating python3 virtual environment. Make sure python3 and pyvenv-3 are installed"
fi


