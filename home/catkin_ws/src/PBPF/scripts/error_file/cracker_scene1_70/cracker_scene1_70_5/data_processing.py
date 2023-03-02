# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""

import ssl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
flag_pos = True
flag_ang = True
flag_CVPF = False
correct_time_flag = False
update_style_flag = "time"
task_flag = "1"
rosbag_flag = "5"
loop_flag = 10
if task_flag == "1":
    if update_style_flag == "pose":
        prepare_time = 1900
    else:
        prepare_time = 3000
elif task_flag == "2":
    if update_style_flag == "pose":
        prepare_time = 1900
    else:
        prepare_time = 2000
elif task_flag == "3":
    if update_style_flag == "pose":
        prepare_time = 1900
    else:
        prepare_time = 3900
else:
    if update_style_flag == "time":
        prepare_time = 2300
test = "cracker_" # cracker_/fish_can_

# cracker_scene1_rosbag1_repeat1_time_obse_err_ang
# cracker_scene1_rosbag1_repeat2_time_obse_err_ang
# cracker_scene1_rosbag1_repeat1_time_obse_err_pos
# 9cracker_time_scene1_obse_err_pos  
obse_pos_name = update_style_flag+'_obse_err_pos.csv'
obse_ang_name = update_style_flag+'_obse_err_ang.csv'
PBPF_pos_name = update_style_flag+'_PBPF_err_pos.csv'
PBPF_ang_name = update_style_flag+'_PBPF_err_ang.csv'
CVPF_pos_name = update_style_flag+'_CVPF_err_pos.csv'
CVPF_ang_name = update_style_flag+'_CVPF_err_ang.csv'
name_before = test+'scene'+task_flag+'_rosbag'+rosbag_flag+'_repeat'
# file_name_PFPE_pos = test+update_style_flag+'_scene'+task_flag+'_PBPF_err_pos.csv'
# file_name_CVPF_pos = test+update_style_flag+'_scene'+task_flag+'_CVPF_err_pos.csv'
# file_name_obse_ang = test+update_style_flag+'_scene'+task_flag+'_obse_err_ang.csv'
# file_name_PFPE_ang = test+update_style_flag+'_scene'+task_flag+'_PBPF_err_ang.csv'
# file_name_CVPF_ang = test+update_style_flag+'_scene'+task_flag+'_CVPF_err_ang.csv'
file_name_pos = test+update_style_flag+'_scene'+task_flag+'_pos_'+task_time+'.csv'
file_name_ang = test+update_style_flag+'_scene'+task_flag+'_ang_'+task_time+'.csv'
# pos

def correct_time(datasetcopy):
    print("Enter into the correct time function")
    for time_index in range(len(datasetcopy)):
        time = datasetcopy.loc[datasetcopy.index==time_index,'time']
        if datasetcopy.time[time_index] > 2:
            datasetcopy.loc[datasetcopy.index==time_index,'time'] = datasetcopy.loc[datasetcopy.index==time_index,'time'] - 16
    # print(datasetcopy.time)
    
if flag_pos == True:
    print("Ready to integrate the data of pos")
    for j in range(loop_flag):
        dataset = pd.read_csv(name_before+str(j+1)+obse_pos_name)
        dataset.columns=["index","time","error","alg"]
        dataset.time = dataset.time - 4.3
        datasetcopy = copy.deepcopy(dataset)
        newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
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
                                 datasetcopy.loc[newdata.idxmin(),'alg']]
        # print(newdataset.time)
        print("obse_pos ",j)
        newdataset.to_csv(file_name_pos,index=0,header=0,mode='a')
    print("finished")
    for j in range(loop_flag):
        dataset = pd.read_csv(name_before+str(j+1)+PBPF_pos_name)
        dataset.columns=["index","time","error","alg"]
        dataset.time = dataset.time - 4.3
        datasetcopy = copy.deepcopy(dataset)
        newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
        # timedf = dataset['time']
        timestep_list = []
        for timestep in range(prepare_time):
            timestep_list.append(timestep/100.0)
        if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
            correct_time(datasetcopy)
        timedf = datasetcopy['time']
        for i in range(prepare_time):
            newdata = (timedf - timestep_list[int(i)]).abs()
            #print(newdata)
            #print(newdata.idxmin())
            datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[int(i)]
            #datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'error'] = dataset.loc[dataset.index==newdata.idxmin(),'error'] - 0.005
            newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                                 datasetcopy.loc[newdata.idxmin(),'time'],
                                 datasetcopy.loc[newdata.idxmin(),'error'],
                                 datasetcopy.loc[newdata.idxmin(),'alg']]
        print("PFPE_pos ",j)
        newdataset.to_csv(file_name_pos,index=0,header=0,mode='a')
    print("finished")
    if flag_CVPF == True:
        for j in range(loop_flag):     
            dataset = pd.read_csv(str(j+1)+file_name_CVPF_pos)
            dataset.columns=["index","time","error","alg"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
            # timedf = dataset['time']
            timestep_list = []
            for timestep in range(prepare_time):
                timestep_list.append(timestep/100.0)
            if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
                correct_time(datasetcopy)
            timedf = datasetcopy['time']
            for i in range(prepare_time):
                newdata = (timedf - timestep_list[int(i)]).abs()
                #print(newdata)
                #print(newdata.idxmin())
                datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[int(i)]
                newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                                    datasetcopy.loc[newdata.idxmin(),'time'],
                                    datasetcopy.loc[newdata.idxmin(),'error'],
                                    datasetcopy.loc[newdata.idxmin(),'alg']]
            print("CVPF_pos ",j)
            newdataset.to_csv(file_name_pos,index=0,header=0,mode='a')
    print("finished")
# ang
if flag_ang == True:
    print("Ready to integrate the data of ang")
    for j in range(loop_flag):
        dataset = pd.read_csv(name_before+str(j+1)+obse_ang_name)
        dataset.columns=["index","time","error","alg"]
        dataset.time = dataset.time - 4.3
        datasetcopy = copy.deepcopy(dataset)
        newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
        # timedf = dataset['time']
        timestep_list = []
        for timestep in range(prepare_time):
            timestep_list.append(timestep/100.0)
        if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
            correct_time(datasetcopy)
        timedf = datasetcopy['time']
        for i in range(prepare_time):
            newdata = (timedf - timestep_list[i]).abs()
            #print(newdata)
            #print(newdata.idxmin())
            datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
            #datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'error'] = dataset.loc[dataset.index==newdata.idxmin(),'error'] + 0.05
            newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                                 datasetcopy.loc[newdata.idxmin(),'time'],
                                 datasetcopy.loc[newdata.idxmin(),'error'],
                                 datasetcopy.loc[newdata.idxmin(),'alg']]
        print("obse_ang ",j)
        newdataset.to_csv(file_name_ang,index=0,header=0,mode='a')
    print("finished")
    for j in range(loop_flag):
        dataset = pd.read_csv(name_before+str(j+1)+PBPF_ang_name)
        dataset.columns=["index","time","error","alg"]
        dataset.time = dataset.time - 4.3
        datasetcopy = copy.deepcopy(dataset)
        newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
        # timedf = dataset['time']
        timestep_list = []
        for timestep in range(prepare_time):
            timestep_list.append(timestep/100.0)
        if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
            correct_time(datasetcopy)
        timedf = datasetcopy['time']
        for i in range(prepare_time):
            newdata = (timedf - timestep_list[i]).abs()
            #print(newdata)
            #print(newdata.idxmin())
            datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
            newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                                 datasetcopy.loc[newdata.idxmin(),'time'],
                                 datasetcopy.loc[newdata.idxmin(),'error'],
                                 datasetcopy.loc[newdata.idxmin(),'alg']]
        print("PFPE_ang ",j)
        newdataset.to_csv(file_name_ang,index=0,header=0,mode='a')
    print("finished")
    if flag_CVPF == True:
        for j in range(loop_flag):     
            dataset = pd.read_csv(str(j+1)+file_name_CVPF_ang)
            dataset.columns=["index","time","error","alg"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
            # timedf = dataset['time']
            timestep_list = []
            for timestep in range(prepare_time):
                timestep_list.append(timestep/100.0)
            if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
                correct_time(datasetcopy)
            timedf = datasetcopy['time']
            for i in range(prepare_time):
                newdata = (timedf - timestep_list[i]).abs()
                #print(newdata)
                #print(newdata.idxmin())
                datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
                #datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'error'] = dataset.loc[dataset.index==newdata.idxmin(),'error'] + 0.05
                newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                                    datasetcopy.loc[newdata.idxmin(),'time'],
                                    datasetcopy.loc[newdata.idxmin(),'error'],
                                    datasetcopy.loc[newdata.idxmin(),'alg']]
            print("CVPF_ang ",j)
            newdataset.to_csv(file_name_ang,index=0,header=0,mode='a')
    print("finished")



'''hue = 'alg','''
#figure = sns.lineplot(x="time", y=newdataset.error,data=newdataset, ci=95)
#svg_fig = figure.get_figure()
#svg_fig.savefig("0_2_scene1a_err_pos.svg",format="svg")
