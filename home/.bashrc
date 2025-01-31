PS1="[trackObjectWithPF] Singularity> \w \$ "
# export ROS_MASTER_URI=http://localhost:11311
export ROS_MASTER_URI=http://panda-workstation:11311
# export ROS_HOSTNAME=server3
export ROS_HOSTNAME=$(hostname)
# export ROS_HOSTNAME=panda-server
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/zisongxu/.local/lib/python3.8/site-packages/nvidia/cuda_nvcc

export PYTHONPATH=/home/zisongxu/pyvkdepth/bin:$PYTHONPATH
