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
task_flag = "3"
update_style_flag = "time"

if update_style_flag == "pose":
    title_name = "Pose"
elif update_style_flag == "time":
    title_name = "Time"

file_name_ang = update_style_flag+"_scene"+task_flag+"_ang"
file_name_pos = update_style_flag+"_scene"+task_flag+"_pos"
title_ang = "Comparison of Rotation Errors Based on "+title_name+" Update in Scene "+task_flag
title_pos = "Comparison of Position Errors Based on "+title_name+" Update in Scene "+task_flag

if flag_plot_ang == True:
    print("Ready to plot the figure of ang")
    dataset_ang = pd.read_csv(file_name_ang+'.csv')
    dataset_ang.columns=["index","time","ang_error","alg"]
    figure_ang = sns.lineplot(x="time", y="ang_error", data=dataset_ang, hue = 'alg', ci=95)
    svg_fig_ang = figure_ang.get_figure()
    plt.title(title_ang)
    svg_fig_ang.savefig(file_name_ang+".svg",format="svg")
if flag_plot_pos ==True:
    print("Ready to plot the figure of pos")
    dataset_pos = pd.read_csv(file_name_pos+'.csv')
    dataset_pos.columns=["index","time","pos_error","alg"]
    figure_pos = sns.lineplot(x="time", y="pos_error", data=dataset_pos, hue = 'alg', ci=95)
    svg_fig_pos = figure_pos.get_figure()
    plt.title(title_pos)
    svg_fig_pos.savefig(file_name_pos+".svg",format="svg")
