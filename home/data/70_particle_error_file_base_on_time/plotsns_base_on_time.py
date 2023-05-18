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


particle_num = sys.argv[1]
object_name = sys.argv[2]
sceneName = sys.argv[3] # "scene1"
update_style_flag = sys.argv[4] # time/pose
ang_and_pos = sys.argv[5] # pos/ang


update_style_flag = "time"
test = "cracker_" # cracker_/fish_can_
if update_style_flag == "pose":
    title_name = "Pose"
elif update_style_flag == "time":
    title_name = "Time"
# 70_cracker_scene1_time_pos.csv
file_name = "based_on_time_"+str(particle_num)+'_'+object_name+'_'+sceneName+'_'+update_style_flag+'_'+ang_and_pos
file_name_ang = test+update_style_flag+"_scene"+sceneName+"_ang"
file_name_pos = test+update_style_flag+"_scene"+sceneName+"_pos"
title_ang = "Rotational errors (rad) vs Time (s)"
title_pos = "Positional errors (m) vs Time (s)"

if ang_and_pos == "ang":
    if sceneName == "scene1":
        x_range_max = 28
        x_range_unit = 2
        y_range_max = math.pi
        y_range_unit = 0.4
        x_xlim = 28
        y_ylim = 3.2
    if sceneName == "scene2":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = math.pi
        y_range_unit = 0.4
        x_xlim = 32
        y_ylim = 3.2
    if sceneName == "scene3":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = math.pi
        y_range_unit = 0.4
        x_xlim = 32
        y_ylim = 3.2
    if sceneName == "scene4":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = 0.4
        y_range_unit = 0.05
        x_xlim = 32
        y_ylim = 0.4
    print("Ready to plot the figure of ang")
    dataset_ang = pd.read_csv(file_name+'.csv', header=None)
    dataset_ang.columns=["index","time","Rotational Error (rad)","alg","obj_scene","particle_num","ray_type"]
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
    svg_fig_ang.savefig(file_name+".svg",format="svg")

if ang_and_pos == "pos":
    if sceneName == "scene1":
        x_range_max = 28
        x_range_unit = 2
        y_range_max = 0.5
        y_range_unit = 0.04
        x_xlim = 28
        y_ylim = 0.5
    if sceneName == "scene2":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = 0.4
        y_range_unit = 0.05
        x_xlim = 32
        y_ylim = 0.5
    if sceneName == "scene3":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = 0.4
        y_range_unit = 0.05
        x_xlim = 32
        y_ylim = 0.5
    if sceneName == "scene4":
        x_range_max = 30
        x_range_unit = 2
        y_range_max = 0.05
        y_range_unit = 0.005
        x_xlim = 32
        y_ylim = 0.05
    print("Ready to plot the figure of pos")
    ymax = 0.12
    dataset_pos = pd.read_csv(file_name+'.csv', header=None)
    dataset_pos.columns=["index","time","Positional Error (m)","alg","obj_scene","particle_num","ray_type"]
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
    svg_fig_pos.savefig(file_name+".svg",format="svg")

print("finished")
