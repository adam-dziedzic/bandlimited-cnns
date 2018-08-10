#!/usr/bin/env bash

mkdir code
cd code
git clone https://github.com/adam-dziedzic/time-series-ml.git  # provide the password
cd time-series-ml
git config credential.helper store  # remember credentials
git pull  # repeat the password