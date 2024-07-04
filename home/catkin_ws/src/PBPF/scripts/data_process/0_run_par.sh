#!/bin/bash

# echo "Running split_data.sh"
# ./split_data.sh &
# wait  # 等待所有后台进程完成
# echo "split_data.sh completed"

# sleep 60

# echo "Running align_time_par_data_process.sh"
# ./align_time_par_data_process.sh &
# wait  # 等待所有后台进程完成
# echo "align_time_par_data_process.sh completed"

# sleep 60

echo "Running compute_par_min_err_after_align.sh"
./compute_par_min_err_after_align.sh &
wait  # 等待所有后台进程完成
echo "compute_par_min_err_after_align.sh completed"

echo "Running compute_par_min_err_after_align.sh"
./compute_par_avg_err_after_align.sh &
wait  # 等待所有后台进程完成
echo "compute_par_avg_err_after_align.sh completed"

# sleep 60

# echo "Running plotsns_new_min_par.sh"
# ./plotsns_new_min_par.sh &
# wait  # 等待所有后台进程完成
# echo "plotsns_new_min_par.sh completed"
