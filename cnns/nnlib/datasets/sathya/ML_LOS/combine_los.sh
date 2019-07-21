#!/usr/bin/env bash

los_type='LOS'
for case in 'TRAIN' 'TEST'; do
    out = 2_classes_WiFi_${case}
    rm -f $out

    touch $out

    cat 6F_${los_type}/2_classes_WiFi_${case} 10F_${los_type}/2_classes_WiFi_${case} 15F_${los_type}/2_classes_WiFi_${case} > $out
done;



