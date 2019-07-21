#!/usr/bin/env bash

for case in TRAIN TEST; do
    cat ../ML_LOS/2_classes_WiFi_LOS_${case} ../ML_NLOS/2_classes_WiFi_NLOS_${case} > 2_classes_WiFi_NLOS_LOS_${case}
done;
