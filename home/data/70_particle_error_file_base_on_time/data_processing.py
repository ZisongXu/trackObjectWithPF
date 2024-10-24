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


particle_num = sys.argv[1]
object_name = sys.argv[2]
task_flag = sys.argv[3] # "scene1"
rosbag_flag = sys.argv[4]
repeat_time = sys.argv[5]
run_alg_flag = sys.argv[6] # PBPF
ang_and_pos = sys.argv[7] # pos/ang


# 1_cracker_scene1_rosbag1_repeat8_time_PBPF_err_pos
file_name = str(particle_num)+'_'+object_name+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+update_style_flag+'_'+run_alg_flag+'_err_'+ang_and_pos
# PBPF_pos_file_name = '_'+update_style_flag+'_PBPF_err_pos.csv'
# PBPF_ang_file_name = '_'+update_style_flag+'_PBPF_err_ang.csv'
# obse_pos_file_name = '_'+update_style_flag+'_obse_err_pos.csv'
# obse_ang_file_name = '_'+update_style_flag+'_obse_err_ang.csv'
# CVPF_pos_file_name = '_'+update_style_flag+'_CVPF_err_pos.csv'
# CVPF_ang_file_name = '_'+update_style_flag+'_CVPF_err_ang.csv'

flag_pos = True
flag_ang = True
flag_CVPF = True
correct_time_flag = False

loop_flag = 10
prepare_time = 2800

# save file
save_file_name = 'based_on_time_'+str(particle_num)+'_'+object_name+'_'+task_flag+'_'+update_style_flag+'_'+ang_and_pos+'.csv'
# save_file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/error_file_diff_par_num/70/1_cracker_scene1/inter_data_"+ang_and_pos+"/")
save_file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/error_file_02_18_new_rosbag1/")

def correct_time(datasetcopy):
    print("Enter into the correct time function")
    for time_index in range(len(datasetcopy)):
        time = datasetcopy.loc[datasetcopy.index==time_index,'time']
        if datasetcopy.time[time_index] > 2:
            datasetcopy.loc[datasetcopy.index==time_index,'time'] = datasetcopy.loc[datasetcopy.index==time_index,'time'] - 16
    # print(datasetcopy.time)

# print("Ready to integrate the data of "+ang_and_pos)
dataset = pd.read_csv(file_name+'.csv', header=None)
# dataset.columns=["index","time","error","alg","obj_scene","particle_num","ray_type"]
dataset.columns=['index','time','error','alg','obj_scene','particle_num','ray_type']
# dataset.time = dataset.time - 4.3
datasetcopy = copy.deepcopy(dataset)
newdataset = pd.DataFrame(columns=['step','time','error','alg','obj_scene','particle_num','ray_type'],index=[])
timestep_list = []
for timestep in range(prepare_time):
    timestep_list.append(timestep/100.0)
if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
    correct_time(datasetcopy)
timedf = datasetcopy['time']
# print(datasetcopy)
# datasetcopy.to_csv("test",index=0,header=0,mode='a')
for i in range(prepare_time):
    newdata = (timedf - timestep_list[int(i)]).abs()
    #print(newdata)
    #print(newdata.idxmin())
    datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[int(i)]
    newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                            datasetcopy.loc[newdata.idxmin(),'time'],
                            datasetcopy.loc[newdata.idxmin(),'error'],
                            datasetcopy.loc[newdata.idxmin(),'alg'],
                            datasetcopy.loc[newdata.idxmin(),'obj_scene'],
                            datasetcopy.loc[newdata.idxmin(),'particle_num'],
                            datasetcopy.loc[newdata.idxmin(),'ray_type']]
# print(newdataset.time)
print(str(particle_num)+'_'+object_name+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+run_alg_flag+'_'+ang_and_pos)
newdataset.to_csv(save_file_path+save_file_name, index=0, header=0, mode='a')

# particle_num = sys.argv[1]
# object_name = sys.argv[2]
# task_flag = sys.argv[3] # "scene1"
# rosbag_flag = sys.argv[4]
# repeat_time = sys.argv[5]
# run_alg_flag = sys.argv[6] # PBPF
# ang_and_pos = sys.argv[7] # pos/ang


#     for j in range(loop_flag):
#         dataset = pd.read_csv(before_name+str(j+1)+PBPF_pos_file_name, header=None)
#         dataset.columns=["index","time","error","alg"]
#         # dataset.time = dataset.time - 4.3
#         datasetcopy = copy.deepcopy(dataset)
#         newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
#         # timedf = dataset['time']
#         timestep_list = []
#         for timestep in range(prepare_time):
#             timestep_list.append(timestep/100.0)
#         if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
#             correct_time(datasetcopy)
#         timedf = datasetcopy['time']
#         for i in range(prepare_time):
#             newdata = (timedf - timestep_list[int(i)]).abs()
#             #print(newdata)
#             #print(newdata.idxmin())
#             datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[int(i)]
#             #datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'error'] = dataset.loc[dataset.index==newdata.idxmin(),'error'] - 0.005
#             newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
#                                  datasetcopy.loc[newdata.idxmin(),'time'],
#                                  datasetcopy.loc[newdata.idxmin(),'error'],
#                                  datasetcopy.loc[newdata.idxmin(),'alg']]
#         print("PFPE_pos ",j)
#         newdataset.to_csv(file_path_pos+file_name_pos,index=0,header=0,mode='a')
#     print("finished")
#     if flag_CVPF == True:
#         for j in range(loop_flag):     
#             dataset = pd.read_csv(before_name+str(j+1)+CVPF_pos_file_name, header=None)
#             dataset.columns=["index","time","error","alg"]
#             datasetcopy = copy.deepcopy(dataset)
#             newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
#             # timedf = dataset['time']
#             timestep_list = []
#             for timestep in range(prepare_time):
#                 timestep_list.append(timestep/100.0)
#             if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
#                 correct_time(datasetcopy)
#             timedf = datasetcopy['time']
#             for i in range(prepare_time):
#                 newdata = (timedf - timestep_list[int(i)]).abs()
#                 #print(newdata)
#                 #print(newdata.idxmin())
#                 datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[int(i)]
#                 newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
#                                     datasetcopy.loc[newdata.idxmin(),'time'],
#                                     datasetcopy.loc[newdata.idxmin(),'error'],
#                                     datasetcopy.loc[newdata.idxmin(),'alg']]
#             print("CVPF_pos ",j)
#             newdataset.to_csv(file_path_pos+file_name_pos,index=0,header=0,mode='a')
#     print("finished")
# # ang
# if flag_ang == True:
#     print("Ready to integrate the data of ang")
#     for j in range(loop_flag):
#         dataset = pd.read_csv(before_name+str(j+1)+obse_ang_file_name, header=None)
#         dataset.columns=["index","time","error","alg"]
#         # dataset.time = dataset.time - 4.3
#         datasetcopy = copy.deepcopy(dataset)
#         newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
#         # timedf = dataset['time']
#         timestep_list = []
#         for timestep in range(prepare_time):
#             timestep_list.append(timestep/100.0)
#         if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
#             correct_time(datasetcopy)
#         timedf = datasetcopy['time']
#         for i in range(prepare_time):
#             newdata = (timedf - timestep_list[i]).abs()
#             #print(newdata)
#             #print(newdata.idxmin())
#             datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
#             #datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'error'] = dataset.loc[dataset.index==newdata.idxmin(),'error'] + 0.05
#             newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
#                                  datasetcopy.loc[newdata.idxmin(),'time'],
#                                  datasetcopy.loc[newdata.idxmin(),'error'],
#                                  datasetcopy.loc[newdata.idxmin(),'alg']]
#         print("obse_ang ",j)
#         newdataset.to_csv(file_path_ang+file_name_ang,index=0,header=0,mode='a')
#     print("finished")
#     for j in range(loop_flag):
#         dataset = pd.read_csv(before_name+str(j+1)+PBPF_ang_file_name, header=None)
#         dataset.columns=["index","time","error","alg"]
#         # dataset.time = dataset.time - 4.3
#         datasetcopy = copy.deepcopy(dataset)
#         newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
#         # timedf = dataset['time']
#         timestep_list = []
#         for timestep in range(prepare_time):
#             timestep_list.append(timestep/100.0)
#         if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
#             correct_time(datasetcopy)
#         timedf = datasetcopy['time']
#         for i in range(prepare_time):
#             newdata = (timedf - timestep_list[i]).abs()
#             #print(newdata)
#             #print(newdata.idxmin())
#             datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
#             newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
#                                  datasetcopy.loc[newdata.idxmin(),'time'],
#                                  datasetcopy.loc[newdata.idxmin(),'error'],
#                                  datasetcopy.loc[newdata.idxmin(),'alg']]
#         print("PFPE_ang ",j)
#         newdataset.to_csv(file_path_ang+file_name_ang,index=0,header=0,mode='a')
#     print("finished")
#     if flag_CVPF == True:
#         for j in range(loop_flag):     
#             dataset = pd.read_csv(before_name+str(j+1)+CVPF_ang_file_name, header=None)
#             dataset.columns=["index","time","error","alg"]
#             datasetcopy = copy.deepcopy(dataset)
#             newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
#             # timedf = dataset['time']
#             timestep_list = []
#             for timestep in range(prepare_time):
#                 timestep_list.append(timestep/100.0)
#             if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
#                 correct_time(datasetcopy)
#             timedf = datasetcopy['time']
#             for i in range(prepare_time):
#                 newdata = (timedf - timestep_list[i]).abs()
#                 #print(newdata)
#                 #print(newdata.idxmin())
#                 datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
#                 #datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'error'] = dataset.loc[dataset.index==newdata.idxmin(),'error'] + 0.05
#                 newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
#                                     datasetcopy.loc[newdata.idxmin(),'time'],
#                                     datasetcopy.loc[newdata.idxmin(),'error'],
#                                     datasetcopy.loc[newdata.idxmin(),'alg']]
#             print("CVPF_ang ",j)
#             newdataset.to_csv(file_path_ang+file_name_ang,index=0,header=0,mode='a')
#     print("finished")



# '''hue = 'alg','''
# #figure = sns.lineplot(x="time", y=newdataset.error,data=newdataset, ci=95)
# #svg_fig = figure.get_figure()
# #svg_fig.savefig("0_2_scene1a_err_pos.svg",format="svg")
