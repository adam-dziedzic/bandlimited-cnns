#!/usr/bin/env bash
rm -rf complex_mul_cuda.egg-info/
rm -rf build/
rm -rf dist/
rm -rf /local/${USER}/anaconda3/lib/python3.6/site-packages/complex_mul_cuda-0.0.0-py3.6-linux-x86_64.egg
rm -rf ~/anaconda3/lib/python3.6/site-packages/complex_mul_cuda-0.0.0-py3.6-linux-x86_64.egg
~/anaconda3/bin/python3.6 setup.py install
# python setup.py install