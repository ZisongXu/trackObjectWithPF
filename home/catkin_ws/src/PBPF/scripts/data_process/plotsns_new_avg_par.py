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

save_file_path = os.path.expanduser('~/catkin_ws/src/PBPF/scripts/results/particles/')

particle_num = sys.argv[1]
object_name = sys.argv[2]
sceneName = sys.argv[3] # "scene1"
rosbag_flag = sys.argv[4] # 8
update_style_flag = sys.argv[5] # time/pose
ang_and_pos = sys.argv[6] # pos/ang/ADD/ADDS
# tem_name = sys.argv[6]

update_style_flag = "time"
test = "cracker_" # cracker_/fish_can_
if update_style_flag == "pose":
    title_name = "Pose"
elif update_style_flag == "time":
    title_name = "Time"

# based_on_time_10_scene1_time_cracker_ADD
file_name = "based_on_time_"+str(particle_num)+'_'+sceneName+'_rosbag'+str(rosbag_flag)+'_'+update_style_flag+'_'+object_name+'_'+ang_and_pos

title_ang = "Rotational errors (rad) vs Time (s)"
title_pos = "Positional errors (m) vs Time (s)"
title_ADD = object_name+": "+ang_and_pos+" errors (m) vs Time (s)"



if ang_and_pos == "ADD" or ang_and_pos == "ADDS" :
    
    
    if object_name == "cracker" and rosbag_flag == "1":
        x_range_max = 30 # 28, 129, 265
        x_range_unit = 3 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.05 # 0.04
        x_xlim = 30 # 28
        y_ylim = 0.5 # 0.5
    if object_name == "Ketchup" and rosbag_flag == "1":
        x_range_max = 100 # 28, 129, 265
        x_range_unit = 10 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.05 # 0.04
        x_xlim = 100 # 28
        y_ylim = 0.5 # 0.5
    if object_name == "Mayo" and rosbag_flag == "1":
        x_range_max = 120 # 28, 129, 265
        x_range_unit = 12 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.05 # 0.04
        x_xlim = 120 # 28
        y_ylim = 0.5 # 0.5
    if object_name == "Milk" and rosbag_flag == "1":
        x_range_max = 85 # 28, 129, 265
        x_range_unit = 9 # 2, 6, 25, 125
        y_range_max = 0.10 # 0.5
        y_range_unit = 0.01 # 0.04
        x_xlim = 85 # 28
        y_ylim = 0.1 # 0.5
    if object_name == "Mustard" and rosbag_flag == "1":
        x_range_max = 120 # 28, 129, 265
        x_range_unit = 12 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.05 # 0.04
        x_xlim = 120 # 28
        y_ylim = 0.5 # 0.5
    if object_name == "Parmesan" and rosbag_flag == "1":
        x_range_max = 85 # 28, 129, 265
        x_range_unit = 9 # 2, 6, 25, 125
        y_range_max = 0.1 # 0.5
        y_range_unit = 0.01 # 0.04
        x_xlim = 85 # 28
        y_ylim = 0.1 # 0.5
    if object_name == "SaladDressing" and rosbag_flag == "1":
        x_range_max = 120 # 28, 129, 265
        x_range_unit = 12 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.05 # 0.04
        x_xlim = 120 # 28
        y_ylim = 0.5 # 0.5
    if object_name == "soup" and rosbag_flag == "1":
        x_range_max = 60 # 28, 129, 265
        x_range_unit = 6 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.025 # 0.04
        x_xlim = 95 # 28
        y_ylim = 0.25 # 0.5
    if object_name == "soup2" and rosbag_flag == "1":
        x_range_max = 60 # 28, 129, 265
        x_range_unit = 6 # 2, 6, 25, 125
        y_range_max = 0.5 # 0.5
        y_range_unit = 0.025 # 0.04
        x_xlim = 95 # 28
        y_ylim = 0.25 # 0.5


    print("Ready to plot the figure of "+ang_and_pos+" ("+object_name+")")
    ymax = 0.12
    dataset_ADD = pd.read_csv(save_file_path+file_name+'_par_avg.csv', header=None)

    print(dataset_ADD)
    if ang_and_pos == "ADD":
        dataset_ADD.columns=["index","time","alg","obj","scene","particle_num","ray_type","obj_name", "mass", "friction","ADD Error (m)"]
    if ang_and_pos == "ADDS":
        dataset_ADD.columns=["index","time","alg","obj","scene","particle_num","ray_type","obj_name", "mass", "friction","ADDS Error (m)"]
    print(pd.__version__)
    print(sns.__version__)
    print(matplotlib.__version__)
    color_map = {
        "FOUD": "#FC8002",
        "DOPE": "#F0EEBB",
        "PBPF_RGBD_par_avg": "#614099",
        "PBPF_RGB_par_avg": "#EE4431",
        "PBPF_D_par_avg": "#369F2D",
        "Diff-DOPE": "#4995C6",
        "Diff-DOPE-Tracking": "#EDB11A",
    }

    # color_map = {
    #     "mA": "#614099",
    #     "mBNN": "#EE4431",
    #     "mC": "#369F2D",
    # }

    # color_map = {
    #     "fA": "#614099",
    #     "fB": "#EE4431",
    #     "fC": "#369F2D",
    #     "fD": "#4995C6",
    # }
    # print("Before")
    # print(dataset_ADD)
    # dataset_ADD = dataset_ADD.to_numpy()[:,np.newaxis]
    # print("After")
    # print(dataset_ADD)
    if ang_and_pos == "ADD":
        figure_ADD = sns.lineplot(data=dataset_ADD, x="time", y="ADD Error (m)", hue='alg', errorbar=('ci', 95), legend=True, linewidth=0.5, palette=color_map)
        # figure_ADD = sns.lineplot(data=dataset_ADD, x="time", y="ADD Error (m)", hue='friction', errorbar=('ci', 95), legend=True, linewidth=0.5, palette=color_map)
    if ang_and_pos == "ADDS":
        figure_ADD = sns.lineplot(data=dataset_ADD, x="time", y="ADDS Error (m)", hue='alg', errorbar=('ci', 95), legend=True, linewidth=0.5, palette=color_map)
        # figure_ADD = sns.lineplot(data=dataset_ADD, x="time", y="ADDS Error (m)", hue='friction', errorbar=('ci', 95), legend=True, linewidth=0.5, palette=color_map)
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
    svg_fig_ADD.savefig(save_file_path+file_name+"_par_avg.svg",format="svg")
    # svg_fig_ADD.savefig(save_file_path+file_name+".svg", format="svg", dpi=150)

    print("finished "+object_name+" par avg "+ang_and_pos+" plot")