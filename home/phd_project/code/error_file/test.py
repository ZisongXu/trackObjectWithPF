# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
flag_pos = True
flag_ang = True
flag_PFPM = True
update_style_flag = "pose"
task_flag = "1b"
loop_flag = 10
if task_flag == "1a":
    if update_style_flag == "pose":
        prepare_time = 1900
    else:
        prepare_time = 1900
elif task_flag == "1b":
    if update_style_flag == "pose":
        prepare_time = 1900
    else:
        prepare_time = 1950
elif task_flag == "2":
    if update_style_flag == "pose":
        prepare_time = 1900
    else:
        prepare_time = 2100
else:
    if update_style_flag == "time":
        prepare_time = 1600

file_name_obse_pos = update_style_flag+'_scene'+task_flag+'_obse_err_pos.csv'
file_name_PFPE_pos = update_style_flag+'_scene'+task_flag+'_PFPE_err_pos.csv'
file_name_PFPM_pos = update_style_flag+'_scene'+task_flag+'_PFPM_err_pos.csv'
file_name_obse_ang = update_style_flag+'_scene'+task_flag+'_obse_err_ang.csv'
file_name_PFPE_ang = update_style_flag+'_scene'+task_flag+'_PFPE_err_ang.csv'
file_name_PFPM_ang = update_style_flag+'_scene'+task_flag+'_PFPM_err_ang.csv'
file_name_pos = update_style_flag+'_scene'+task_flag+'_pos.csv'
file_name_ang = update_style_flag+'_scene'+task_flag+'_ang.csv'
# pos
if flag_pos == True:
    print("Ready to integrate the data of pos")
    for j in range(loop_flag):
        dataset = pd.read_csv(str(j+1)+file_name_obse_pos)
        dataset.columns=["index","time","error","alg"]
        datasetcopy = copy.deepcopy(dataset)
        newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
        timedf = dataset['time']
        timestep_list = []
        for timestep in range(prepare_time):
            timestep_list.append(timestep/100.0)
        for i in range(prepare_time):
            newdata = (timedf - timestep_list[int(i)]).abs()
            #print(newdata)
            #print(newdata.idxmin())
            datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[int(i)]
            newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                                 datasetcopy.loc[newdata.idxmin(),'time'],
                                 datasetcopy.loc[newdata.idxmin(),'error'],
                                 datasetcopy.loc[newdata.idxmin(),'alg']]
        print("obse_pos ",j)
        newdataset.to_csv(file_name_pos,index=0,header=0,mode='a')
    print("finished")
    for j in range(loop_flag):
        dataset = pd.read_csv(str(j+1)+file_name_PFPE_pos)
        dataset.columns=["index","time","error","alg"]
        datasetcopy = copy.deepcopy(dataset)
        newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
        timedf = dataset['time']
        timestep_list = []
        for timestep in range(prepare_time):
            timestep_list.append(timestep/100.0)
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
    if flag_PFPM == True:
        for j in range(loop_flag):     
            dataset = pd.read_csv(str(j+1)+file_name_PFPM_pos)
            dataset.columns=["index","time","error","alg"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
            timedf = dataset['time']
            timestep_list = []
            for timestep in range(prepare_time):
                timestep_list.append(timestep/100.0)
            for i in range(prepare_time):
                newdata = (timedf - timestep_list[int(i)]).abs()
                #print(newdata)
                #print(newdata.idxmin())
                datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[int(i)]
                newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                                    datasetcopy.loc[newdata.idxmin(),'time'],
                                    datasetcopy.loc[newdata.idxmin(),'error'],
                                    datasetcopy.loc[newdata.idxmin(),'alg']]
            print("PFPM_pos ",j)
            newdataset.to_csv(file_name_pos,index=0,header=0,mode='a')
    print("finished")
# ang
if flag_ang == True:
    print("Ready to integrate the data of ang")
    for j in range(loop_flag):
        dataset = pd.read_csv(str(j+1)+file_name_obse_ang)
        dataset.columns=["index","time","error","alg"]
        datasetcopy = copy.deepcopy(dataset)
        newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
        timedf = dataset['time']
        timestep_list = []
        for timestep in range(prepare_time):
            timestep_list.append(timestep/100.0)
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
        dataset = pd.read_csv(str(j+1)+file_name_PFPE_ang)
        dataset.columns=["index","time","error","alg"]
        datasetcopy = copy.deepcopy(dataset)
        newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
        timedf = dataset['time']
        timestep_list = []
        for timestep in range(prepare_time):
            timestep_list.append(timestep/100.0)
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
    if flag_PFPM == True:
        for j in range(loop_flag):     
            dataset = pd.read_csv(str(j+1)+file_name_PFPM_ang)
            dataset.columns=["index","time","error","alg"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
            timedf = dataset['time']
            timestep_list = []
            for timestep in range(prepare_time):
                timestep_list.append(timestep/100.0)
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
            print("PFPM_ang ",j)
            newdataset.to_csv(file_name_ang,index=0,header=0,mode='a')
    print("finished")



'''hue = 'alg','''
#figure = sns.lineplot(x="time", y=newdataset.error,data=newdataset, ci=95)
#svg_fig = figure.get_figure()
#svg_fig.savefig("0_2_scene1a_err_pos.svg",format="svg")
