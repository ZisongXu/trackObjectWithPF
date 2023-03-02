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
flag_CVPF = True
correct_time_flag = False
update_style_flag = "time"
task_flag = "2"
task_time = "1"
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
        prepare_time = 3000
elif task_flag == "3":
    if update_style_flag == "pose":
        prepare_time = 1900
    else:
        prepare_time = 3900
else:
    if update_style_flag == "time":
        prepare_time = 2300
test = "soup_" # cracker_/fish_can_/soup_
file_name_obse_pos = test+update_style_flag+'_scene'+task_flag+'_obse_err_pos.csv'
file_name_PFPE_pos = test+update_style_flag+'_scene'+task_flag+'_PBPF_err_pos.csv'
file_name_CVPF_pos = test+update_style_flag+'_scene'+task_flag+'_CVPF_err_pos.csv'
file_name_obse_ang = test+update_style_flag+'_scene'+task_flag+'_obse_err_ang.csv'
file_name_PFPE_ang = test+update_style_flag+'_scene'+task_flag+'_PBPF_err_ang.csv'
file_name_CVPF_ang = test+update_style_flag+'_scene'+task_flag+'_CVPF_err_ang.csv'
file_name_pos = test+update_style_flag+'_scene'+task_flag+'_pos'
file_name_ang = test+update_style_flag+'_scene'+task_flag+'_ang'
# pos

def correct_time(datasetcopy):
    print("Enter into the correct time function")
    for time_index in range(len(datasetcopy)):
        time = datasetcopy.loc[datasetcopy.index==time_index,'time']
        if datasetcopy.time[time_index] > 2:
            datasetcopy.loc[datasetcopy.index==time_index,'time'] = datasetcopy.loc[datasetcopy.index==time_index,'time'] - 16
    # print(datasetcopy.time)
    
# ang
if flag_ang == True:
    print("Ready to integrate the data of ang")
    dataset = pd.read_csv(file_name_ang+'_ori.csv')
    # dataset = pd.read_csv('test.csv')
    dataset.columns=["index","time","error","alg"]
    dataset.error = dataset.error - 0.8
    dataset.to_csv(file_name_ang+'.csv',index=0,header=0,mode='a')
    # dataset.to_csv('tezt1.csv',index=0,header=0,mode='a')
    



'''hue = 'alg','''
#figure = sns.lineplot(x="time", y=newdataset.error,data=newdataset, ci=95)
#svg_fig = figure.get_figure()
#svg_fig.savefig("0_2_scene1a_err_pos.svg",format="svg")
