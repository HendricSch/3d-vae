#!/bin/bash

pip install virtualenv

virtualenv venv

source venv/bin/activate

pip install -r requirements_nvidia.txt

