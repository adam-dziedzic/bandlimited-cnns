#!/usr/bin/env bash
for i in 1024 256 128 64 32 16 8 4 2; do
scp -r NLOS-6-${i}/6_classes_WIFI cc@129.114.108.29:~/code/bandlimited-cnns/cnns/nnlib/datasets/sathya/data_journal/NLOS-6-${i}/;
done