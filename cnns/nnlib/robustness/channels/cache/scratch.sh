#!/usr/bin/env bash

for i in adversarial_image*
do
    mv "$i" "${i/.npy.npy/.npy}"
done