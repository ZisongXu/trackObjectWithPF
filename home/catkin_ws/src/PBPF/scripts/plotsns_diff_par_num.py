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
import sys

flag_plot = sys.argv[1]# "pos" #pos/ang

task_flag = "1"
update_style_flag = "time"
test = "cracker_" # cracker_/fish_can_
if update_style_flag == "pose":
    title_name = "Pose"
elif update_style_flag == "time":
    title_name = "Time"

file_name_ang = test+update_style_flag+"_scene"+task_flag+"_ang"
file_name_pos = test+update_style_flag+"_scene"+task_flag+"_pos"
title_ang = "Rotational errors (rad) vs Time (s)"
title_pos = "Positional errors (m) vs Time (s)"

if flag_plot == "ang":
    if task_flag == "1":
        x_range_max = 32
        x_range_unit = 5
        y_range_max = 2.0
        y_range_unit = 0.2
        x_xlim = 32
        y_ylim = 2.0

    print("Ready to plot the figure of ang")
    dataset_ang = pd.read_csv(flag_plot+'.csv')
    dataset_ang.columns=["index","time","Rotational Error (rad)","alg","obj_scene","particle_num"]
    figure_ang = sns.lineplot(x="particle_num", y="Rotational Error (rad)", data=dataset_ang, hue = 'obj_scene', errorbar=('ci', 95), legend=True, linewidth = 0.5)
    figure_ang.set(xlabel = None, ylabel = None)
    # figure_ang.set_xlabel(None)
    x = range(0, x_range_max, x_range_unit)
    y = np.arange(0, y_range_max, y_range_unit)
    plt.xticks(x)
    plt.yticks(y)
    plt.tick_params(labelsize=15)
    plt.xlim(0, x_xlim)
    plt.ylim(0, y_ylim)
    plt.title(title_ang, fontsize=16)
    svg_fig_ang = figure_ang.get_figure()
    svg_fig_ang.savefig(file_name_ang+".svg",format="svg")

if flag_plot == "pos" or flag_plot == "all":
    if task_flag == "1":
        x_range_max = 32
        x_range_unit = 5
        y_range_max = 0.27
        y_range_unit = 0.025
        x_xlim = 32
        y_ylim = 0.30

    print("Ready to plot the figure of pos")
    ymax = 0.12
    dataset_pos = pd.read_csv(flag_plot+'.csv')
    dataset_pos.columns=["index","time","Positional Error (m)","alg","obj_scene","particle_num"]
    figure_pos = sns.lineplot(x="particle_num", y="Positional Error (m)", data=dataset_pos, hue = 'obj_scene', errorbar=('ci', 95), legend=True, linewidth = 0.5, err_style = "bars")
    figure_pos.set(xlabel = None, ylabel = None)
    x = range(0, x_range_max, x_range_unit)
    y = np.arange(0, y_range_max, y_range_unit)
    plt.xticks(x)
    plt.yticks(y)
    plt.tick_params(labelsize=15)
    plt.xlim(0, x_xlim)
    plt.ylim(0, y_ylim)
    plt.title(title_pos, fontsize=16)
    svg_fig_pos = figure_pos.get_figure()
    svg_fig_pos.savefig(file_name_pos+".svg",format="svg")

print("finished")
