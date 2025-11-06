# vqa-framework
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package contains utilities for processing, testing, and loading CLEVR-like datasets.

### Setting up the Environment in your new repo:

First create a python3.7 virtual env: `python3.7 -m venv env`

Next install requirements.txt from this repo: `source env/bin/activate` then `pip install -r requirements.txt`

Finally, `pip install -e ./vqa-framework`.

NOTE: This "package" doesn't currently declare any dependencies. 
This was a deliberate choice as the code will
probably work with other versions of libraries outside of requirements.txt,
but there are no guarantees. Probably best to run the test suite if you do.

## CUDA:

requirements.txt file assume `CUDA 10.2` and `cudnn-10.2-v7.6.5`.
Also currently using Python3.7