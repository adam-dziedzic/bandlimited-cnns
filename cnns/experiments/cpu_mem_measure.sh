#!/usr/bin/env bash
energy_preserved=${1:-90}  # either the 1st argument value of energy preserved 90%
batch_size=${2:-256}
epochs=${3:-1}

echo 3 | sudo tee /proc/sys/vm/drop_caches;
TIMESTAMP=$(date -d"$CURRENT +$MINUTES minutes" '+%F-%H-%M-%S-%N-%Z')
out_file="standard_compression_preserve"${energy_preserved}"_percent_"${TIMESTAMP}".txt"
echo "batch_size: "${batch_size} >> ${out_file}
echo "epochs: "${epochs} >> ${out_file}

sar -b -d -q -r -u -W -o ${out_file}.sar 1 > /dev/null 2>&1 &
SAR_PID=$!

sleep 3

START=$(date +%s.%N)
PYTHONPATH=/home/${USER}/code/time-series-ml nohup ~/anaconda3/bin/python -m memory_profiler /home/${USER}/code/time-series-ml/cnns/nnlib/pytorch_timeseries/fcnn_fft.py --conv_type=FFT1D --epochs=${epochs} --index_back=0 --preserve_energy=${energy_preserved} --datasets=debug --lr=0.001 --optimizer_type=ADAM --compress_type=STANDARD --min-batch-size=${batch_size} >> ${out_file} 2>&1
END=$(date +%s.%N)
DIFF_TIME=$(echo "$END - $START" | bc)

echo "diff time: "${DIFF_TIME} >> ${out_file}

sleep 3
kill -9 $SAR_PID # sar blocks on wait
sar -r 1 -f ${out_file}.sar >>  ${out_file}.sar.csv
