#!/bin/bash
module load python/3.6.3
virtualenv ~/ENV
source ~/ENV/bin/activate
python setup.py install
