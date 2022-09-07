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
flag_pos = True
flag_ang = True
flag_DOPE = True
flag_PFPE = True
flag_PFPM = True
update_style_flag = "pose"
task_flag = "1b"
loop_flag = 10

file_name_obse_pos = update_style_flag+'_scene'+task_flag+'_obse_err_pos.csv'
file_name_PFPE_pos = update_style_flag+'_scene'+task_flag+'_PFPE_err_pos.csv'
file_name_PFPM_pos = update_style_flag+'_scene'+task_flag+'_PFPM_err_pos.csv'
file_name_obse_ang = update_style_flag+'_scene'+task_flag+'_obse_err_ang.csv'
file_name_PFPE_ang = update_style_flag+'_scene'+task_flag+'_PFPE_err_ang.csv'
file_name_PFPM_ang = update_style_flag+'_scene'+task_flag+'_PFPM_err_ang.csv'
file_name_pos = update_style_flag+'_scene'+task_flag+'_pos.csv'
file_name_ang = update_style_flag+'_scene'+task_flag+'_ang.csv'
error_file_name_DOPE_pos = "pos_error_DOPE"+"_scene"+task_flag
error_file_name_DOPE_ang = "ang_error_DOPE"+"_scene"+task_flag
error_file_name_PFPE_pos = "pos_error_PFPE"+"_scene"+task_flag
error_file_name_PFPE_ang = "ang_error_PFPE"+"_scene"+task_flag
error_file_name_PFPM_pos = "pos_error_PFPM"+"_scene"+task_flag
error_file_name_PFPM_ang = "ang_error_PFPM"+"_scene"+task_flag
# pos
if flag_pos == True:
    print("Ready to generate the data of pos")
    if flag_DOPE == True:
        print("Ready to generate the data of pos in DOPE")
        pos_error_all_list = []
        for j in range(loop_flag):
            dataset = pd.read_csv(str(j+1)+file_name_obse_pos)
            dataset.columns=["index","time","error","alg"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','error','alg'],index=[])
            # timedf = dataset['time']
            pos_error = dataset['error']
            for index in range(len(pos_error)):
                pos_error_all_list.append(datasetcopy.error[index])
        print("DOPE pos error mean:",np.mean(pos_error_all_list))
        print("DOPE pos error  std:",np.std(pos_error_all_list))
    if flag_PFPE == True:
        print("Ready to generate the data of pos in PFPE")
        pos_error_all_list = []
        for j in range(loop_flag):
            dataset = pd.read_csv(str(j+1)+file_name_PFPE_pos)
            dataset.columns=["index","time","error","alg"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','error','alg'],index=[])
            # timedf = dataset['time']
            pos_error = dataset['error']
            for index in range(len(pos_error)):
                pos_error_all_list.append(datasetcopy.error[index])
        print("PFPE pos error mean:",np.mean(pos_error_all_list))
        print("PFPE pos error  std:",np.std(pos_error_all_list))
    if flag_PFPM == True:
        print("Ready to generate the data of pos in PFPM")
        pos_error_all_list = []
        for j in range(loop_flag):
            dataset = pd.read_csv(str(j+1)+file_name_PFPM_pos)
            dataset.columns=["index","time","error","alg"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','error','alg'],index=[])
            # timedf = dataset['time']
            pos_error = dataset['error']
            for index in range(len(pos_error)):
                pos_error_all_list.append(datasetcopy.error[index])
        print("PFPM pos error mean:",np.mean(pos_error_all_list))
        print("PFPM pos error  std:",np.std(pos_error_all_list))
# ang
if flag_ang == True:
    print("Ready to integrate the data of ang")
    if flag_DOPE == True:
        print("Ready to generate the data of ang in DOPE")
        ang_error_all_list = []
        for j in range(loop_flag):
            dataset = pd.read_csv(str(j+1)+file_name_obse_ang)
            dataset.columns=["index","time","error","alg"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','error','alg'],index=[])
            # timedf = dataset['time']
            ang_error = dataset['error']
            for index in range(len(ang_error)):
                ang_error_all_list.append(datasetcopy.error[index])
        print("DOPE ang error mean:",np.mean(ang_error_all_list))
        print("DOPE ang error  std:",np.std(ang_error_all_list))
    if flag_PFPE == True:
        print("Ready to generate the data of ang in PFPE")
        ang_error_all_list = []
        for j in range(loop_flag):
            dataset = pd.read_csv(str(j+1)+file_name_PFPE_ang)
            dataset.columns=["index","time","error","alg"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','error','alg'],index=[])
            # timedf = dataset['time']
            ang_error = dataset['error']
            for index in range(len(ang_error)):
                ang_error_all_list.append(datasetcopy.error[index])
        print("PFPE ang error mean:",np.mean(ang_error_all_list))
        print("PFPE ang error  std:",np.std(ang_error_all_list))
    if flag_PFPM == True:
        print("Ready to generate the data of ang in PFPM")
        ang_error_all_list = []
        for j in range(loop_flag):
            dataset = pd.read_csv(str(j+1)+file_name_PFPM_ang)
            dataset.columns=["index","time","error","alg"]
            datasetcopy = copy.deepcopy(dataset)
            newdataset = pd.DataFrame(columns=['step','time','error','alg'],index=[])
            # timedf = dataset['time']
            ang_error = dataset['error']
            for index in range(len(ang_error)):
                ang_error_all_list.append(datasetcopy.error[index])
        print("PFPM ang error mean:",np.mean(ang_error_all_list))
        print("PFPM ang error  std:",np.std(ang_error_all_list))


'''hue = 'alg','''
#figure = sns.lineplot(x="time", y=newdataset.error,data=newdataset, ci=95)
#svg_fig = figure.get_figure()
#svg_fig.savefig("0_2_scene1a_err_pos.svg",format="svg")
