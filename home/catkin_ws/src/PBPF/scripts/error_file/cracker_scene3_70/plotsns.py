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
import os
import yaml

with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
    parameter_info = yaml.safe_load(file)
object_name_list = parameter_info['object_name_list']
object_flag = object_name_list[0]
task_flag = parameter_info['task_flag'] #1/2/3/4  
test = object_flag+"_" # cracker_/fish_can_
update_style_flag = parameter_info['update_style_flag']

flag_plot_ang = True
flag_plot_pos = False


if update_style_flag == "pose":
    title_name = "Pose"
elif update_style_flag == "time":
    title_name = "Time"

file_name_ang = test+update_style_flag+"_scene"+task_flag+"_ang"
file_name_pos = test+update_style_flag+"_scene"+task_flag+"_pos"
title_ang = "Rotational errors (rad) vs Time (s)"
title_pos = "Positional errors (m) vs Time (s)"

if flag_plot_ang == True:
    if task_flag == "1":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = math.pi
        y_range_unit = 0.4
        x_xlim = 32
        y_ylim = 3.2
    if task_flag == "2":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = math.pi
        y_range_unit = 0.4
        x_xlim = 32
        y_ylim = 3.2
    if task_flag == "3":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = math.pi
        y_range_unit = 0.4
        x_xlim = 32
        y_ylim = 3.2
    print("Ready to plot the figure of ang")
    dataset_ang = pd.read_csv(file_name_ang+'.csv')
    dataset_ang.columns=["index","time","Rotational Error (rad)","alg"]
    figure_ang = sns.lineplot(x="time", y="Rotational Error (rad)", data=dataset_ang, hue = 'alg', errorbar=('ci', 95), legend=False, linewidth = 0.5)
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
if flag_plot_pos == True:
    if task_flag == "1":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = 0.4
        y_range_unit = 0.05
        x_xlim = 32
        y_ylim = 0.5
    if task_flag == "2":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = 0.4
        y_range_unit = 0.05
        x_xlim = 32
        y_ylim = 0.5
    if task_flag == "3":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = 0.4
        y_range_unit = 0.05
        x_xlim = 32
        y_ylim = 0.5
    print("Ready to plot the figure of pos")
    ymax = 0.12
    dataset_pos = pd.read_csv(file_name_pos+'.csv')
    dataset_pos.columns=["index","time","Positional Error (m)","alg"]
    figure_pos = sns.lineplot(x="time", y="Positional Error (m)", data=dataset_pos, hue = 'alg', errorbar=('ci', 95), legend=False, linewidth = 0.5)
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
