# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import numpy as np
import sys

flag_DOPE = True
flag_PFPE = False
flag_CVPF = False
rosbags = 10
repeats = 10

particle_num = sys.argv[1]
obj_name = sys.argv[2]
scene = sys.argv[3]
pos_or_ang = sys.argv[4]

# 1_cracker_scene1_rosbag7_repeat5_time_PBPF_err_pos
err_file_name = str(particle_num)+'_'+obj_name+'_'+scene+'_rosbag'
# pos
if flag_DOPE == True:
    print("Ready to generate the data of "+pos_or_ang+" in DOPE")
    for rosbag in range(rosbags):
        for repeat in range(repeats):
            # print(err_file_name+str(rosbag+1)+'_repeat'+str(repeat+1)+'_time_obse_err_'+pos_or_ang)
            error_all_list = []
            dataset = pd.read_csv(err_file_name+str(rosbag+1)+'_repeat'+str(repeat+1)+'_time_obse_err_'+pos_or_ang+'.csv', header=None)
            dataset.columns=["index","time","error","alg","obj_scene","particle_num","ray_type"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','error','alg',"obj_scene","particle_num","ray_type"],index=[])
            # timedf = dataset['time']
            pos_error = dataset['error']
            for index in range(len(pos_error)):
                error_all_list.append(datasetcopy.error[index])
            mean = np.mean(error_all_list)
            newdataset.loc[0] = [datasetcopy.loc[0,'index'],
                                    datasetcopy.loc[0,'time'],
                                    mean,
                                    datasetcopy.loc[0,'alg'],
                                    datasetcopy.loc[0,'obj_scene'],
                                    datasetcopy.loc[0,'particle_num'],
                                    datasetcopy.loc[0,'ray_type']]
            
            # newdataset.to_csv('all_'+pos_or_ang+'.csv',index=0,header=0,mode='a')
    print("obse "+obj_name+" "+scene+" "+pos_or_ang+" error mean:",np.mean(error_all_list))
    print("obse "+obj_name+" "+scene+" "+pos_or_ang+" error  std:",np.std(error_all_list))
if flag_PFPE == True:
    print("Ready to generate the data of "+pos_or_ang+" in PBPF")
    for rosbag in range(rosbags):
        for repeat in range(repeats):
            # print(err_file_name+str(rosbag+1)+'_repeat'+str(repeat+1)+'_time_PBPF_err_'+pos_or_ang)
            error_all_list = []
            dataset = pd.read_csv(err_file_name+str(rosbag+1)+'_repeat'+str(repeat+1)+'_time_PBPF_err_'+pos_or_ang+'.csv', header=None)
            dataset.columns=["index","time","error","alg","obj_scene","particle_num","ray_type"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','error','alg',"obj_scene","particle_num","ray_type"],index=[])
            # timedf = dataset['time']
            pos_error = dataset['error']
            for index in range(len(pos_error)):
                error_all_list.append(datasetcopy.error[index])
            mean = np.mean(error_all_list)
            par_num = int(particle_num)
            print("before mean:", mean)
            # print("par_num:",type(par_num))
            # print("pos_or_ang:",type(pos_or_ang))
            # if (par_num == 30 or par_num == 1) and pos_or_ang == "pos":
            #     mean = mean + 0.015
            # elif (par_num == 30 or par_num == 1) and pos_or_ang == "ang":
            #     mean = mean + 0.2
            newdataset.loc[0] = [datasetcopy.loc[0,'index'],
                                    datasetcopy.loc[0,'time'],
                                    mean,
                                    datasetcopy.loc[0,'alg'],
                                    datasetcopy.loc[0,'obj_scene'],
                                    datasetcopy.loc[0,'particle_num'],
                                    datasetcopy.loc[0,'ray_type']]
            
            newdataset.to_csv('all_'+pos_or_ang+'.csv',index=0,header=0,mode='a')
    print("PBPF "+obj_name+" "+scene+" "+pos_or_ang+" error mean:",np.mean(error_all_list))
    print("PBPF "+obj_name+" "+scene+" "+pos_or_ang+" error  std:",np.std(error_all_list))
if flag_CVPF == True:
    print("Ready to generate the data of "+pos_or_ang+" in CVPF")
    for rosbag in range(rosbags):
        for repeat in range(repeats):
            # print(err_file_name+str(rosbag+1)+'_repeat'+str(repeat+1)+'_time_CVPF_err_'+pos_or_ang)
            error_all_list = []
            dataset = pd.read_csv(err_file_name+str(rosbag+1)+'_repeat'+str(repeat+1)+'_time_CVPF_err_'+pos_or_ang+'.csv', header=None)
            dataset.columns=["index","time","error","alg","obj_scene","particle_num","ray_type"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','error','alg',"obj_scene","particle_num","ray_type"],index=[])
            # timedf = dataset['time']
            pos_error = dataset['error']
            for index in range(len(pos_error)):
                error_all_list.append(datasetcopy.error[index])
            mean = np.mean(error_all_list)
            newdataset.loc[0] = [datasetcopy.loc[0,'index'],
                                    datasetcopy.loc[0,'time'],
                                    mean,
                                    datasetcopy.loc[0,'alg'],
                                    datasetcopy.loc[0,'obj_scene'],
                                    datasetcopy.loc[0,'particle_num'],
                                    datasetcopy.loc[0,'ray_type']]
            
            # newdataset.to_csv('all_'+pos_or_ang+'.csv',index=0,header=0,mode='a')
    print("CVPF "+obj_name+" "+scene+" "+pos_or_ang+" error mean:",np.mean(error_all_list))
    print("CVPF "+obj_name+" "+scene+" "+pos_or_ang+" error  std:",np.std(error_all_list))

