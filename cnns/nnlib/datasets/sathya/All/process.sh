#!/usr/bin/env bash

for case in TRAIN TEST; do
cat ../AllCases/WIFI_AllCases_${case} ../ML_NLOS_LOS/2_classes_WiFi_LOS_NLOS_${case} > 2_classes_WiFi_all_${case}
done;