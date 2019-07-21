#!/usr/bin/env bash

los_type='NLOS'
for case in 'TRAIN' 'TEST'; do
    rm -f 2_classes_WiFi_${case}

    touch 2_classes_WiFi_${los_type}_${case}

    cat 6F_${los_type}/2_classes_WiFi_${case} 10F_${los_type}/2_classes_WiFi_${case} 15F_${los_type}/2_classes_WiFi_${case} > 2_classes_WiFi_${los_type}_${case}
done;