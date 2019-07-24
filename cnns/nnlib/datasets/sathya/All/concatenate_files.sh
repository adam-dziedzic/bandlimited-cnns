#!/usr/bin/env bash

for f in *.txt; do (cat "${f}"; echo) >> final; done