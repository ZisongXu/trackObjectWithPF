# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy


flag_plot_ang = True
flag_plot_pos = False
file_name_ang = "dis_scene1a_ang"
file_name_pos = "dis_scene1a_pos"
if flag_plot_ang == True:
    print("Ready to plot the figure of ang")
    dataset_ang = pd.read_csv(file_name_ang+'.csv')
    dataset_ang.columns=["index","time","ang_error","alg"]
    figure_ang = sns.lineplot(x="time", y="ang_error", data=dataset_ang, hue = 'alg', ci=95)
    svg_fig_ang = figure_ang.get_figure()
    plt.title("Comparison of Rotation Errors Based on Distance Update in Scene 1a")
    svg_fig_ang.savefig(file_name_ang+".svg",format="svg")
if flag_plot_pos ==True:
    print("Ready to plot the figure of pos")
    dataset_pos = pd.read_csv(file_name_pos+'.csv')
    dataset_pos.columns=["index","time","pos_error","alg"]
    figure_pos = sns.lineplot(x="time", y="pos_error", data=dataset_pos, hue = 'alg', ci=95)
    svg_fig_pos = figure_pos.get_figure()
    plt.title("Comparison of Position Errors Based on Distance Update in Scene 1a")
    svg_fig_pos.savefig(file_name_pos+".svg",format="svg")
