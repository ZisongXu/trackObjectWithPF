# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy

# pos
for j in range(25):
    dataset = pd.read_csv(str(j+1)+'02_scene1a_obse_err_pos.csv')
    dataset.columns=["index","time","error","alg"]
    datasetcopy = copy.deepcopy(dataset)
    newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    timedf = dataset['time']
    timestep_list = []
    for timestep in range(301):
        timestep_list.append(timestep/10.0)
    for i in range(301):
        newdata = (timedf - timestep_list[i]).abs()
        #print(newdata)
        #print(newdata.idxmin())
        datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
        newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                             datasetcopy.loc[newdata.idxmin(),'time'],
                             datasetcopy.loc[newdata.idxmin(),'error'],
                             datasetcopy.loc[newdata.idxmin(),'alg']]
    newdataset.to_csv('newdataset_pos.csv',index=0,header=0,mode='a')
print("finished")

for j in range(25):
    dataset = pd.read_csv(str(j+1)+'02_scene1a_PFPE_err_pos.csv')
    dataset.columns=["index","time","error","alg"]
    datasetcopy = copy.deepcopy(dataset)
    newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    timedf = dataset['time']
    timestep_list = []
    for timestep in range(301):
        timestep_list.append(timestep/10.0)
    for i in range(301):
        newdata = (timedf - timestep_list[i]).abs()
        #print(newdata)
        #print(newdata.idxmin())
        datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
        newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                             datasetcopy.loc[newdata.idxmin(),'time'],
                             datasetcopy.loc[newdata.idxmin(),'error'],
                             datasetcopy.loc[newdata.idxmin(),'alg']]
    newdataset.to_csv('newdataset_pos.csv',index=0,header=0,mode='a')
print("finished")
    
for j in range(25):     
    dataset = pd.read_csv(str(j+1)+'02_scene1a_PFPM_err_pos.csv')
    dataset.columns=["index","time","error","alg"]
    datasetcopy = copy.deepcopy(dataset)
    newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    timedf = dataset['time']
    timestep_list = []
    for timestep in range(301):
        timestep_list.append(timestep/10.0)
    for i in range(301):
        newdata = (timedf - timestep_list[i]).abs()
        #print(newdata)
        #print(newdata.idxmin())
        datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
        newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                             datasetcopy.loc[newdata.idxmin(),'time'],
                             datasetcopy.loc[newdata.idxmin(),'error'],
                             datasetcopy.loc[newdata.idxmin(),'alg']]
    newdataset.to_csv('newdataset_pos.csv',index=0,header=0,mode='a')
print("finished")

# ang
for j in range(25):
    dataset = pd.read_csv(str(j+1)+'02_scene1a_obse_err_ang.csv')
    dataset.columns=["index","time","error","alg"]
    datasetcopy = copy.deepcopy(dataset)
    newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
    timedf = dataset['time']
    timestep_list = []
    for timestep in range(301):
        timestep_list.append(timestep/10.0)
    for i in range(301):
        newdata = (timedf - timestep_list[i]).abs()
        #print(newdata)
        #print(newdata.idxmin())
        datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
        newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                             datasetcopy.loc[newdata.idxmin(),'time'],
                             datasetcopy.loc[newdata.idxmin(),'error'],
                             datasetcopy.loc[newdata.idxmin(),'alg']]
    newdataset.to_csv('newdataset_ang.csv',index=0,header=0,mode='a')
print("finished")


for j in range(25):
    dataset = pd.read_csv(str(j+1)+'02_scene1a_PFPE_err_ang.csv')
    dataset.columns=["index","time","error","alg"]
    datasetcopy = copy.deepcopy(dataset)
    newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
    timedf = dataset['time']
    timestep_list = []
    for timestep in range(301):
        timestep_list.append(timestep/10.0)
    for i in range(301):
        newdata = (timedf - timestep_list[i]).abs()
        #print(newdata)
        #print(newdata.idxmin())
        datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
        newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                             datasetcopy.loc[newdata.idxmin(),'time'],
                             datasetcopy.loc[newdata.idxmin(),'error'],
                             datasetcopy.loc[newdata.idxmin(),'alg']]
    newdataset.to_csv('newdataset_ang.csv',index=0,header=0,mode='a')
print("finished")
    
for j in range(25):     
    dataset = pd.read_csv(str(j+1)+'02_scene1a_PFPM_err_ang.csv')
    dataset.columns=["index","time","error","alg"]
    datasetcopy = copy.deepcopy(dataset)
    newdataset = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
    timedf = dataset['time']
    timestep_list = []
    for timestep in range(301):
        timestep_list.append(timestep/10.0)
    for i in range(301):
        newdata = (timedf - timestep_list[i]).abs()
        #print(newdata)
        #print(newdata.idxmin())
        datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
        newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                             datasetcopy.loc[newdata.idxmin(),'time'],
                             datasetcopy.loc[newdata.idxmin(),'error'],
                             datasetcopy.loc[newdata.idxmin(),'alg']]
    newdataset.to_csv('newdataset_ang.csv',index=0,header=0,mode='a')
print("finished")



'''hue = 'alg','''
#figure = sns.lineplot(x="time", y=newdataset.error,data=newdataset, ci=95)
#svg_fig = figure.get_figure()
#svg_fig.savefig("0_2_scene1a_err_pos.svg",format="svg")
