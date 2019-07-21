#!/usr/bin/env bash

for case in TEST TRAIN; do
    cat CaseA_los_${case} CaseA_nlos_${case} CaseB_los_${case}0 CaseB_nlos_${case}0 CaseC_los_${case}0 CaseC_nlos_${case}0 > WIFI_AllCases_${case}
done;

for case in TEST TRAIN; do
    for los_type in los nlos; do
        cat CaseA_${los_type}_${case} CaseB_${los_type}_${case}0 CaseC_${los_type}_${case}0 > WIFI_AllCases_${los_type}_${case}
    done;
done;
