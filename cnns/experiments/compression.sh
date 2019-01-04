#!/usr/bin/env bash

awk 'NR % 1563 == 0' additional-info-2018-12-29-07-33-19-893444-all-83-energy-preserved.csv > every_epoch_compression_83_percent.csv
