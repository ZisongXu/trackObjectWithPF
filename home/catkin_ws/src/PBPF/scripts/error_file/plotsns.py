# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import math
import numpy as np
flag_plot_ang = True
flag_plot_pos = False
task_flag = "1"
update_style_flag = "time"

if update_style_flag == "pose":
    title_name = "Pose"
elif update_style_flag == "time":
    title_name = "Time"

file_name_ang = update_style_flag+"_scene"+task_flag+"_ang"
file_name_pos = update_style_flag+"_scene"+task_flag+"_pos"
title_ang = "Rotational errors (rad) vs Time (s)"
title_pos = "Positional errors (m) vs Time (s)"

if flag_plot_ang == True:
    print("Ready to plot the figure of ang")
    dataset_ang = pd.read_csv(file_name_ang+'.csv')
    dataset_ang.columns=["index","time","Rotational Error (rad)","alg"]
    figure_ang = sns.lineplot(x="time", y="Rotational Error (rad)", data=dataset_ang, hue = 'alg', errorbar=('ci', 95), legend=False, linewidth = 0.5)
    figure_ang.set(xlabel = None, ylabel = None)
    # figure_ang.set_xlabel(None)
    x = range(0,65,2)
    y = np.arange(0,math.pi/2.0+0.2, 0.2)
    plt.xticks(x)
    plt.yticks(y)
    plt.tick_params(labelsize=15)
    plt.xlim(0,65)
    plt.ylim(0, math.pi / 2.0)
    plt.title(title_ang, fontsize=16)
    svg_fig_ang = figure_ang.get_figure()
    svg_fig_ang.savefig(file_name_ang+".svg",format="svg")
if flag_plot_pos == True:
    print("Ready to plot the figure of pos")
    ymax = 0.4/2.0
    dataset_pos = pd.read_csv(file_name_pos+'.csv')
    dataset_pos.columns=["index","time","Positional Error (m)","alg"]
    figure_pos = sns.lineplot(x="time", y="Positional Error (m)", data=dataset_pos, hue = 'alg', errorbar=('ci', 95), legend=False, linewidth = 0.5)
    figure_pos.set(xlabel = None, ylabel = None)
    x = range(0,65,2)
    y = np.arange(0, 0.2, 0.02)
    plt.xticks(x)
    plt.yticks(y)
    plt.tick_params(labelsize=15)
    plt.xlim(0,65)
    plt.ylim(0,ymax)
    plt.title(title_pos, fontsize=16)
    svg_fig_pos = figure_pos.get_figure()
    svg_fig_pos.savefig(file_name_pos+".svg",format="svg")
