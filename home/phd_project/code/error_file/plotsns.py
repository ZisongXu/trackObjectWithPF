# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy

dataset_ang = pd.read_csv('newdataset_ang.csv')
dataset_ang.columns=["index","time","ang_error","alg"]
figure_ang = sns.lineplot(x="time", y="ang_error", data=dataset_ang, hue = 'alg', ci=95)
svg_fig_ang = figure_ang.get_figure()
svg_fig_ang.savefig("0_2_scene1a_err_ang.svg",format="svg")


dataset_pos = pd.read_csv('newdataset_pos.csv')
dataset_pos.columns=["index","time","pos_error","alg"]
figure_pos = sns.lineplot(x="time", y="pos_error", data=dataset_pos, hue = 'alg', ci=95)
svg_fig_pos = figure_pos.get_figure()
svg_fig_pos.savefig("0_2_scene1a_err_pos.svg",format="svg")
