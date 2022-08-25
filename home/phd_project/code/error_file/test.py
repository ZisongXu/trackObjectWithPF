# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy

for j in range(3):
    dataset = pd.read_csv('02_scene1a_obse_err_pos'+str(j+1)+'.csv')
    datasetcopy = copy.deepcopy(dataset)
    newdataset = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    
    timedf = dataset['time']
    
    timestep_list = []
    for timestep in range(251):
        timestep_list.append(timestep/10.0)
    for i in range(251):
        newdata = (timedf - timestep_list[i]).abs()
        #print(newdata)
        #print(newdata.idxmin())
        datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[i]
        newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                             datasetcopy.loc[newdata.idxmin(),'time'],
                             datasetcopy.loc[newdata.idxmin(),'error'],
                             datasetcopy.loc[newdata.idxmin(),'alg']]
    newdataset.to_csv('newdataset.csv',index=0,header=0,mode='a')

'''hue = 'alg','''
#figure = sns.lineplot(x="time", y=newdataset.error,data=newdataset, ci=95)
#svg_fig = figure.get_figure()
#svg_fig.savefig("0_2_scene1a_err_pos.svg",format="svg")
