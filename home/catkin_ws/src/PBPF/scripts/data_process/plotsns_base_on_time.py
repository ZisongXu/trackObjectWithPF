# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""


# import seaborn as sns
# sns.set_theme(style="darkgrid")

# # Load an example dataset with long-form data
# fmri = sns.load_dataset("fmri")
# print(fmri)
# print(fmri["timepoint"].flatten())
# # Plot the responses for different events and regions
# sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              data=fmri)

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import math
import numpy as np
import sys
import os
import yaml

with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
        parameter_info = yaml.safe_load(file)
object_name = parameter_info['object_name_list'][0]
gazebo_flag = parameter_info['gazebo_flag']
particle_num = parameter_info['particle_num']
# update mode (pose/time)
update_style_flag = parameter_info['update_style_flag'] # time/pose
# which algorithm to run
run_alg_flag = parameter_info['run_alg_flag'] # PBPF/CVPF
# scene
task_flag = parameter_info['task_flag'] # parameter_info['task_flag']
# rosbag_flag = "1"
err_file = parameter_info['err_file']

save_file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/"+err_file+"/")


particle_num = sys.argv[1]
object_name = sys.argv[2]
sceneName = sys.argv[3] # "scene1"
rosbag_flag = sys.argv[4] # 8
update_style_flag = sys.argv[5] # time/pose
ang_and_pos = sys.argv[6] # pos/ang
# tem_name = sys.argv[6]

update_style_flag = "time"
test = "cracker_" # cracker_/fish_can_
if update_style_flag == "pose":
    title_name = "Pose"
elif update_style_flag == "time":
    title_name = "Time"
# 70_cracker_scene1_time_pos.csv
# based_on_time_150_scene3_time_cracker_ang
# based_on_time_150_scene3_rosbag4_time_cracker_ang
# based_on_time_70_scene2_time_cracker_ADD
# based_on_time_70_scene2_time_soup_ADD
file_name = "based_on_time_"+str(particle_num)+'_'+sceneName+'_'+update_style_flag+'_'+object_name+'_'+ang_and_pos

title_ang = "Rotational errors (rad) vs Time (s)"
title_pos = "Positional errors (m) vs Time (s)"
title_ADD = object_name+": ADD Matrix errors (m) vs Time (s)"

if ang_and_pos == "ang":
    if sceneName == "scene1":
        x_range_max = 340
        x_range_unit = 25
        y_range_max = 5
        y_range_unit = 0.4
        x_xlim = 340
        y_ylim = 5
    if sceneName == "scene2":
        x_range_max = 1300
        x_range_unit = 100
        y_range_max = 2.5
        y_range_unit = 0.2
        x_xlim = 1300
        y_ylim = 2.5
    if sceneName == "scene3":
        # x_range_max = 28
        # x_range_unit = 2
        # y_range_max = 2.5
        # y_range_unit = 0.2
        # x_xlim = 28
        # y_ylim = 2.5
        x_range_max = 265
        x_range_unit = 25
        y_range_max = 2.4
        y_range_unit = 0.2
        x_xlim = 265
        y_ylim = 2.4
    if sceneName == "scene4":
        x_range_max = 28
        x_range_unit = 2
        # y_range_max = 0.4
        y_range_max = 2.5
        y_range_unit = 0.2
        x_xlim = 28
        # y_ylim = 0.4
        y_ylim = 2.5
    if sceneName == "scene5":
        # x_range_max = 28
        # x_range_unit = 2
        # y_range_max = 2.5
        # y_range_unit = 0.2
        # x_xlim = 28
        # y_ylim = 2.5
        x_range_max = 3480
        x_range_unit = 300
        y_range_max = 5
        y_range_unit = 0.4
        x_xlim = 3480
        y_ylim = 5
    print("Ready to plot the figure of ang")
    dataset_ang = pd.read_csv(save_file_path+file_name+'.csv', header=None)
    dataset_ang.columns=["index","time","Rotational Error (rad)","alg","obj_scene","particle_num","ray_type","obj_name"]
    figure_ang = sns.lineplot(x="time", y="Rotational Error (rad)", data=dataset_ang, palette=['y', 'g', 'r'], hue = 'alg', errorbar=('ci', 95), legend=True, linewidth = 0.5)
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
    svg_fig_ang.savefig(save_file_path+file_name+".png",format="png")

if ang_and_pos == "pos":
    if sceneName == "scene1":
        x_range_max = 265 # 28, 129, 265
        x_range_unit = 25 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.04 # 0.04
        x_xlim = 265 # 28
        y_ylim = 0.5 # 0.5
    if sceneName == "scene2":
        x_range_max = 1300
        x_range_unit = 100
        y_range_max = 1.5
        y_range_unit = 0.1
        x_xlim = 1300
        y_ylim = 1.5
    if sceneName == "scene3":
        # x_range_max = 28
        # x_range_unit = 2
        # y_range_max = 0.5
        # y_range_unit = 0.04
        # x_xlim = 28
        # y_ylim = 0.5
        x_range_max = 265 # 28, 129, 265, 1
        x_range_unit = 25 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.04 # 0.04
        x_xlim = 265 # 28
        y_ylim = 0.5 # 0.5
    if sceneName == "scene4":
        x_range_max = 28
        x_range_unit = 2
        y_range_max = 0.5
        y_range_unit = 0.04
        x_xlim = 28
        y_ylim = 0.5
    if sceneName == "scene5":
        # x_range_max = 28
        # x_range_unit = 2
        # y_range_max = 0.5
        # y_range_unit = 0.04
        # x_xlim = 28
        # y_ylim = 0.5
        x_range_max = 3480 # 28, 129, 265, 1
        x_range_unit = 300 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.04 # 0.04
        x_xlim = 3480 # 28
        y_ylim = 0.5 # 0.5
    print("Ready to plot the figure of pos")
    ymax = 0.12
    dataset_pos = pd.read_csv(save_file_path+file_name+'.csv', header=None)

    print(dataset_pos)
    dataset_pos.columns=["index","time","Positional Error (m)","alg","obj_scene","particle_num","ray_type","obj_name"]
    print(pd.__version__)
    print(sns.__version__)
    print(matplotlib.__version__)
    # print("Before")
    # print(dataset_pos)
    # dataset_pos = dataset_pos.to_numpy()[:,np.newaxis]
    # print("After")
    # print(dataset_pos)


    figure_pos = sns.lineplot(data=dataset_pos, x="time", y="Positional Error (m)", palette=['y', 'g', 'r'], hue='alg', errorbar=('ci', 95), legend=True, linewidth=0.5)
    # figure_pos = sns.lineplot(data=dataset_pos, x=1, y=2, hue=3, errorbar=('ci', 95), legend=False, linewidth = 0.5)
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
    svg_fig_pos.savefig(save_file_path+file_name+".png",format="png")

if ang_and_pos == "ADD":
    if sceneName == "scene1":
        x_range_max = 1300 # 28, 129, 265
        x_range_unit = 130 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.05 # 0.04
        x_xlim = 1300 # 28
        y_ylim = 0.5 # 0.5
    if sceneName == "scene2":
        x_range_max = 950
        x_range_unit = 95
        y_range_max = 0.5
        y_range_unit = 0.05
        x_xlim = 950
        y_ylim = 0.5
    if sceneName == "scene3":
        # x_range_max = 28
        # x_range_unit = 2
        # y_range_max = 0.5
        # y_range_unit = 0.04
        # x_xlim = 28
        # y_ylim = 0.5
        x_range_max = 265 # 28, 129, 265, 1
        x_range_unit = 25 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.04 # 0.04
        x_xlim = 265 # 28
        y_ylim = 0.5 # 0.5
    if sceneName == "scene4":
        x_range_max = 28
        x_range_unit = 2
        y_range_max = 0.5
        y_range_unit = 0.04
        x_xlim = 28
        y_ylim = 0.5
    if sceneName == "scene5":
        # x_range_max = 28
        # x_range_unit = 2
        # y_range_max = 0.5
        # y_range_unit = 0.04
        # x_xlim = 28
        # y_ylim = 0.5
        x_range_max = 3480 # 28, 129, 265, 1
        x_range_unit = 300 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.04 # 0.04
        x_xlim = 3480 # 28
        y_ylim = 0.5 # 0.5
    print("Ready to plot the figure of ADD")
    ymax = 0.12
    dataset_ADD = pd.read_csv(save_file_path+file_name+'.csv', header=None)

    print(dataset_ADD)
    dataset_ADD.columns=["index","time","ADD Matrix Error (m)","alg","obj_scene","particle_num","ray_type","obj_name"]
    print(pd.__version__)
    print(sns.__version__)
    print(matplotlib.__version__)
    # print("Before")
    # print(dataset_ADD)
    # dataset_ADD = dataset_ADD.to_numpy()[:,np.newaxis]
    # print("After")
    # print(dataset_ADD)

    figure_ADD = sns.lineplot(data=dataset_ADD, x="time", y="ADD Matrix Error (m)", hue='alg', errorbar=('ci', 95), legend=True, linewidth=0.5)
    # figure_ADD = sns.lineplot(data=dataset_ADD, x="time", y="ADD Matrix Error (m)", palette=['y', 'g', 'r'], hue='alg', errorbar=('ci', 95), legend=True, linewidth=0.5)
    # figure_ADD = sns.lineplot(data=dataset_ADD, x=1, y=2, hue=3, errorbar=('ci', 95), legend=False, linewidth = 0.5)
    figure_ADD.set(xlabel = None, ylabel = None)
    x = range(0, x_range_max, x_range_unit)
    y = np.arange(0, y_range_max, y_range_unit)
    plt.xticks(x)
    plt.yticks(y)
    plt.tick_params(labelsize=15)
    plt.xlim(0, x_xlim)
    plt.ylim(0, y_ylim)
    plt.title(title_ADD, fontsize=16)
    svg_fig_ADD = figure_ADD.get_figure()
    svg_fig_ADD.savefig(save_file_path+file_name+".png",format="png")
    # svg_fig_ADD.savefig(save_file_path+file_name+".svg", format="svg", dpi=150)

print("finished")
