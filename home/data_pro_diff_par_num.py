
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
import yaml
import sys

file_name = sys.argv[1]
with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
    parameter_info = yaml.safe_load(file)

update_style_flag = parameter_info['update_style_flag'] # time/pose
run_alg_flag = parameter_info['run_alg_flag'] # PBPF/CVPF
task_flag = parameter_info['task_flag'] # '1/2/3/4' parameter_info['task_flag']
particle_num = parameter_info['particle_num']  
particle_num = parameter_info['particle_num']
object_name_list = parameter_info['object_name_list']
object_name = object_name_list[0]


front_name = str(particle_num)+"_"+object_name+"_scene"+task_flag+"_rosbag"
behind_name_ang = "_"+update_style_flag+"_"+run_alg_flag+"_err_ang"
behind_name_pos = "_"+update_style_flag+"_"+run_alg_flag+"_err_pos"

for rosbag_index in range(10):
    rosbag_index = rosbag_index + 1
    for repeat_index in range(10):
        repeat_index = repeat_index + 1
        file_name_ang = front_name+str(rosbag_index)+'_repeat'+str(repeat_index)+behind_name_ang
        dataset_ang = pd.read_csv(file_name_ang+'.csv')
        dataset_ang.columns=["index","time","Rotational Error (rad)","alg"]
        dataset_ang.insert(loc = 4, column = 'obj_scene', value = object_name+"_scene"+task_flag)
        dataset_ang.insert(loc = 5, column = 'particle_num', value = particle_num)
        dataset_ang.to_csv(file_name_ang+'.csv')

        file_name_pos = front_name+str(rosbag_index)+'_repeat'+str(repeat_index)+behind_name_pos
        dataset_pos = pd.read_csv(file_name_pos+'.csv')
        dataset_pos.columns=["index","time","Positional Error (m)","alg"]
        dataset_pos.insert(loc = 4, column = 'obj_scene', value = object_name+"_scene"+task_flag)
        dataset_pos.insert(loc = 5, column = 'particle_num', value = particle_num)
        dataset_pos.to_csv(file_name_pos+'.csv')



