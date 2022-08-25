# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
dataset = pd.read_csv('newdataset.csv')


'''hue = 'alg','''
figure = sns.lineplot(x="time", y="error",data=dataset, ci=95)
svg_fig = figure.get_figure()
svg_fig.savefig("0_2_scene1a_err_pos.svg",format="svg")
