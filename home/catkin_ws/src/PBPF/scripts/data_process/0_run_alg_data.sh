#!/bin/bash

 
echo "Running align_time_data_process.sh"
./align_time_data_process.sh &
PID1=$!
wait $PID1 # 等待所有后台进程完成
echo "align_time_data_process.sh completed"

sleep 60

echo "Running compute_err_after_align.sh"
./compute_err_after_align.sh &
PID2=$!
wait $PID2 # 等待所有后台进程完成
echo "compute_err_after_align.sh completed"

sleep 60

# echo "Running plotsns_new.sh"
# ./plotsns_new.sh &
# PID3=$!
# wait $PID3 # 等待所有后台进程完成
# echo "plotsns_new.sh completed"

# sleep 60

# echo "Running compute_par_min_err_after_align.sh"
# ./compute_par_min_err_after_align.sh &
# wait  # 等待所有后台进程完成
# echo "compute_par_min_err_after_align.sh completed"

# sleep 60

# echo "Running compute_par_ave_err_after_align.sh"
# ./compute_par_ave_err_after_align.sh &
# wait  # 等待所有后台进程完成
# echo "compute_par_avg_err_after_align.sh completed"

# sleep 60

# echo "Running plotsns_new_min_par.sh"
# ./plotsns_new_min_par.sh &
# wait  # 等待所有后台进程完成
# echo "plotsns_new_min_par.sh completed"




