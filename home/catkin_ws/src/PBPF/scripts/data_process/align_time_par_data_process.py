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

particle_num = sys.argv[1]
object_name = sys.argv[2]
task_flag = sys.argv[3] # "scene1"
rosbag_flag = sys.argv[4]
repeat_time = sys.argv[5]
run_alg_flag = sys.argv[6] # PBPF
ang_and_pos = sys.argv[7] # pos/ang
runVersion = sys.argv[8] # ("PBPF_RGBD" "PBPF_RGB" "PBPF_D")
par_index = sys.argv[9] # multiray/ang
MASS_marker = sys.argv[10] # obj_name
FRICTION_marker = sys.argv[11] # obj_name


# 10_scene1_rosbag1_repeat0_cracker_time_PBPF_pose_PBPF_RGBD
# 70_scene1_rosbag1_repeat0_time_PBPF_pose_PBPF_D_0
# 5_scene1_rosbag1_repeat0_time_PBPF_pose_PBPF_RGBD_0
# 50_scene1_rosbag1_repeat0_time_PBPF_pose_PBPF_RGBD_mA_fA7_Milk.csv
# 50_scene1_rosbag1_repeat4_time_PBPF_pose_PBPF_RGBD_mA_fA_28_Parmesan
file_name = str(particle_num)+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+update_style_flag+'_'+run_alg_flag+'_pose_'+runVersion+'_'+MASS_marker+'_'+FRICTION_marker+'_'+str(par_index)+'_'+object_name
print(file_name)
flag_pos = True
flag_ang = True
flag_CVPF = True
correct_time_flag = False

loop_flag = 10
prepare_time = 28 * 100
prepare_time = 129 * 100
prepare_time = 265 * 100
prepare_time = 1730 * 100
prepare_time = 1600 * 100
rosbag_slowdown_rate = 20

if object_name == "cracker" and rosbag_flag == "1":
    prepare_time = 100 * 100
    rosbag_slowdown_rate = 1
if object_name == "Ketchup" and rosbag_flag == "1":
    prepare_time = 100 * 100
    rosbag_slowdown_rate = 1
if object_name == "Mayo" and rosbag_flag == "1":
    prepare_time = 115 * 100
    rosbag_slowdown_rate = 1
if object_name == "Milk" and rosbag_flag == "1":
    prepare_time = 100 * 100
    rosbag_slowdown_rate = 1
if object_name == "Mustard" and rosbag_flag == "1":
    prepare_time = 115 * 100
    rosbag_slowdown_rate = 1
if object_name == "Parmesan" and rosbag_flag == "1":
    prepare_time = 30 * 100
    rosbag_slowdown_rate = 1
if object_name == "SaladDressing" and rosbag_flag == "1":
    prepare_time = 115 * 100
    rosbag_slowdown_rate = 1
if object_name == "soup" and rosbag_flag == "1":
    prepare_time = 85 * 100
    rosbag_slowdown_rate = 1
if object_name == "soup2" and rosbag_flag == "1":
    prepare_time = 60 * 100
    rosbag_slowdown_rate = 1

# save file
# save_file_name = tem_name+'_'+'based_on_time_'+str(particle_num)+'_'+object_name+'_'+task_flag+'_'+update_style_flag+'_'+ang_and_pos+'.csv'
# save_file_name = 'based_on_time_'+str(particle_num)+'_'+task_flag+'_'+update_style_flag+'_'+object_name+'_'+ang_and_pos+'.csv'
save_file_name = 'Time_aligned_'+file_name+'.csv'

# save_file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/error_file_diff_par_num/70/1_cracker_scene1/inter_data_"+ang_and_pos+"/")
# save_file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/"+err_file+"/")
save_file_path = os.path.expanduser('~/catkin_ws/src/PBPF/scripts/results/particles/')

def correct_time(datasetcopy):
    print("Enter into the correct time function")
    for time_index in range(len(datasetcopy)):
        time = datasetcopy.loc[datasetcopy.index==time_index,'time']
        if datasetcopy.time[time_index] > 2:
            datasetcopy.loc[datasetcopy.index==time_index,'time'] = datasetcopy.loc[datasetcopy.index==time_index,'time'] - 16
    # print(datasetcopy.time)

def angle_correction(angle):
    if math.pi <= angle <= (3.0 * math.pi):
        angle = angle - 2 * math.pi
    elif -(3.0 * math.pi) <= angle <= -math.pi:
        angle = angle + 2 * math.pi
    angle = abs(angle)
    return angle

# print("Ready to integrate the data of "+ang_and_pos)
dataset = pd.read_csv(save_file_path+file_name+'.csv', header=None)
# dataset.columns=["index","time","error","alg","obj_scene","particle_num","ray_type"]
dataset.columns=['index','time','pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w','alg','obj','scene','particle_num','ray_type', 'obj_name', 'mass', 'friction']
# dataset.time = dataset.time - 4.3







datasetcopy = copy.deepcopy(dataset)
newdataset = pd.DataFrame(columns=['step','time','pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w','alg','obj','scene','particle_num','ray_type', 'obj_name', 'mass', 'friction'],index=[])
timestep_list = []
for timestep in range(int(prepare_time/rosbag_slowdown_rate)):
    timestep_list.append(timestep/100.0 * rosbag_slowdown_rate)
if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
    correct_time(datasetcopy)
timedf = datasetcopy['time']
# print(datasetcopy)
# datasetcopy.to_csv("test",index=0,header=0,mode='a')
for i in range(int(prepare_time/rosbag_slowdown_rate)):
    print("Align particle time: "+file_name+" processing... ", i)
    newdata = (timedf - timestep_list[int(i)]).abs()
    #print(newdata)
    #print(newdata.idxmin())
    if object_name == "gelatin" and ang_and_pos == "ang":
        datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[int(i)]
        newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                             datasetcopy.loc[newdata.idxmin(),'time'],
                             angle_correction(datasetcopy.loc[newdata.idxmin(),'error']+math.pi),
                             datasetcopy.loc[newdata.idxmin(),'alg'],
                             datasetcopy.loc[newdata.idxmin(),'obj_scene'],
                             datasetcopy.loc[newdata.idxmin(),'particle_num'],
                             datasetcopy.loc[newdata.idxmin(),'ray_type'],
                             datasetcopy.loc[newdata.idxmin(),'obj_name']]
    else:
        datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[int(i)]
        newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                             datasetcopy.loc[newdata.idxmin(),'time'],
                             datasetcopy.loc[newdata.idxmin(),'pos_x'],
                             datasetcopy.loc[newdata.idxmin(),'pos_y'],
                             datasetcopy.loc[newdata.idxmin(),'pos_z'],
                             datasetcopy.loc[newdata.idxmin(),'ori_x'],
                             datasetcopy.loc[newdata.idxmin(),'ori_y'],
                             datasetcopy.loc[newdata.idxmin(),'ori_z'],
                             datasetcopy.loc[newdata.idxmin(),'ori_w'],
                             datasetcopy.loc[newdata.idxmin(),'alg'],
                             datasetcopy.loc[newdata.idxmin(),'obj'],
                             datasetcopy.loc[newdata.idxmin(),'scene'],
                             datasetcopy.loc[newdata.idxmin(),'particle_num'],
                             datasetcopy.loc[newdata.idxmin(),'ray_type'],
                             datasetcopy.loc[newdata.idxmin(),'obj_name'],
                             datasetcopy.loc[newdata.idxmin(),'mass'],
                             datasetcopy.loc[newdata.idxmin(),'friction']]
# print(newdataset.time)
# print(str(particle_num)+'_'+object_name+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+run_alg_flag+'_'+ang_and_pos)
print("Done")
newdataset.to_csv(save_file_path+save_file_name, index=0, header=0, mode='w')

