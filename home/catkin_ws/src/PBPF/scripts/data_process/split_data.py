# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""
# gazebo_flag: false
# object_name_list:
# - soup
# - fish_can
# object_num: 1
# other_obj_num: 0
# oto_name_list:
# - base_of_cracker
# - fish_can
# particle_num: 140
# robot_num: 1
# run_alg_flag: PBPF
# task_flag: '3'
# update_style_flag: time
import ssl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import yaml
import os
import sys
import math

with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
        parameter_info = yaml.safe_load(file)
object_name = parameter_info['object_name_list'][0]
gazebo_flag = parameter_info['gazebo_flag']
particle_num = parameter_info['particle_num']
# update mode (pose/time)
update_style_flag = parameter_info['update_style_flag'] # time/pose
# which algorithm to run
run_alg_flag = parameter_info['run_alg_flag'] # PBPF/CVPF
# scene
task_flag = parameter_info['task_flag'] # parameter_info['task_flag']
# rosbag_flag = "1"
err_file = parameter_info['err_file']
# NEW MARKER
MASS_marker = parameter_info['MASS_marker']
FRICTION_marker = parameter_info['FRICTION_marker']

particle_num = sys.argv[1]
task_flag = sys.argv[2] # "scene1"
rosbag_flag = sys.argv[3]
repeat_time = sys.argv[4]
run_alg_flag = sys.argv[5] # PBPF
runVersion = sys.argv[6] # multiray/ang
par_index = sys.argv[7] # multiray/ang
obj_name_path = sys.argv[8] # obj_name
MASS_marker = sys.argv[9] # obj_name
FRICTION_marker = sys.argv[10] # obj_name

# 10_scene1_rosbag1_repeat0_cracker_time_PBPF_pose_PBPF_RGBD
# 70_scene1_rosbag1_repeat0_time_PBPF_pose_PBPF_D_0
# 5_scene1_rosbag1_repeat0_time_PBPF_pose_PBPF_RGBD_0
# 5_scene1_rosbag1_repeat1_time_PBPF_pose_PBPF_RGBD_PBPF_RGBD
# 50_scene1_rosbag1_repeat0_time_PBPF_pose_PBPF_RGBD_mA_fA_0
file_name = str(particle_num)+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+update_style_flag+'_'+run_alg_flag+'_pose_'+runVersion+'_'+MASS_marker+'_'+FRICTION_marker+'_'+str(par_index)
# save_file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/error_file_diff_par_num/70/1_cracker_scene1/inter_data_"+ang_and_pos+"/")
# save_file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/"+err_file+"/")
# /particles/70_rosbag1_repeat0
load_file_path = os.path.expanduser('~/catkin_ws/src/PBPF/scripts/results/particles/'+obj_name_path+'/')
save_file_path = os.path.expanduser('~/catkin_ws/src/PBPF/scripts/results/particles/')
# save_file_path = os.path.expanduser('~/catkin_ws/src/PBPF/scripts/results/particles/')



# print("Ready to integrate the data of "+ang_and_pos)
dataset = pd.read_csv(load_file_path+file_name+'.csv', header=None)
# dataset.columns=["index","time","error","alg","obj_scene","particle_num","ray_type"]
dataset.columns=['index','time','pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w','alg','obj','scene','particle_num','ray_type', 'obj_name', 'mass', 'friction']
# dataset.time = dataset.time - 4.3

# 根据 'obj_name' 分组数据
grouped_datasets = {name: group for name, group in dataset.groupby('obj_name')}

for object_name, group in grouped_datasets.items():
    group.to_csv(save_file_path+file_name+'_'+object_name+'.csv', index=0, header=0, mode='w')
    print(file_name+'_'+object_name+'.csv')

print("Done!")
