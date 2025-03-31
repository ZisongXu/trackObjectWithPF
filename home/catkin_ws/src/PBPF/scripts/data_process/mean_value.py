# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""


import ssl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps
import seaborn as sns
import copy
import yaml
import os
import sys
import math

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


normal_and_par_list = ["par", "normal"] # par/normal
normal_and_par_list = ["normal_3"] # par/normal/Opti_RGB/normal_quick_push/normal_scene2_clear/normal_quick_push_obstacle/normal_scene1_obstacle/same_objects_pushig/normal_1/normal_2/normal_3
ang_and_pos_list = ["ADD", "ADDS"]
# ang_and_pos_list = ["ADD"]

# normal_and_par_list = ["par"] # par/normal
# ang_and_pos_list = ["ADD"]



ang_and_pos_list_len = len(ang_and_pos_list)

error_thresholds = [0.001 * i for i in range(0, 101)]



columns_name = ['step', 'time', 'alg', 'obj', 'scene', 'particle_num', 'ray_type', 'obj_name', 'Errors']
# columns_name = ['step', 'time', 'alg', 'obj', 'scene', 'particle_num', 'ray_type', 'obj_name', 'mass', 'friction', 'Errors']
for ang_and_pos in ang_and_pos_list:
    for normal_and_par in normal_and_par_list:
        panda_data_list = []
        file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/results/0_error_all/"+normal_and_par+"/"+ang_and_pos+"/")
        txt_file_count = len([file for file in os.listdir(file_path) if file.endswith('.csv')])
        print("file_path:", file_path)
        # for q in range(task_flag_list_len):
        #     for w in range(object_name_list_len):
        #         for e in range(ang_and_pos_list_len):
        #             # based_on_time_70_scene1_time_Mustard_ADD
        
        for csv_index in range(txt_file_count):
            file_name = str(csv_index+1)+".csv"
            # print("file_name:", file_name)
            data = pd.read_csv(file_path+file_name, names=columns_name, header=None)
            panda_data_list.append(data)

        combined_data = pd.concat(panda_data_list)
        error_means = combined_data.groupby('alg')['Errors'].mean()
        average_errors_combined = combined_data.groupby(['obj_name', 'alg'])['Errors'].mean().reset_index()

        print("error_means:", error_means)
        print("average_errors_combined:", average_errors_combined)

        result = {}
        for alg, group in combined_data.groupby('alg'):
            total_count = len(group)
            proportions = {}
            for threshold in error_thresholds:
                count_below_threshold = len(group[group['Errors'] < threshold])
                proportions[f'Errors < {threshold}'] = count_below_threshold / total_count
            result[alg] = proportions
        result_df = pd.DataFrame(result).transpose()
        print("result_df:")
        print(result_df)




        color_map = {
            "FOUD": "#FC8002",
            "DOPE": "#F0EEBB",
            "PBPF_RGBD_par_min": "#614099",
            "PBPF_RGBD_par_avg": "#614099",
            "PBPF_RGB_par_min": "#EE4431",
            "PBPF_RGB_par_avg": "#EE4431",
            "PBPF_D_par_min": "#369F2D",
            "PBPF_D_par_avg": "#369F2D",
            "Diff-DOPE": "#4995C6",
            "Diff-DOPE-Tracking": "#EDB11A",
            "PBPF_Opti_par_min": "#000000",
            "PBPF_Opti_par_avg": "#000000",
        }



        error_thresholds = [0.001 * i for i in range(0, 101)]
        x_values = [0] + error_thresholds
        
        # Plot the data
        plt.figure(figsize=(10, 6))

        label__labels = []
        for label_ in result_df.index:
            y_values = [0] + result_df.loc[label_].values.tolist()

            # if label_ == "PBPF_RGBD_par_avg" or label_ == "FOUD" or label_ == "Diff-DOPE" or label_ == "PBPF_Opti_par_avg" or label_ == "PBPF_RGB_par_avg":
            if label_ == "PBPF_RGBD_par_avg" or label_ == "FOUD" or label_ == "Diff-DOPE" or label_ == "PBPF_Opti_par_avg":
                plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'))
                # continue
            else:
                continue
            # elif label_ == "mAN" or label_ == "mBN" or label_ == "mCN" or label_ == "mDN" or label_ == "mEN" or label_ == "mFN" or label_ == "mGN" :# or 
            #     plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle='--')
            #     # continue
            # elif label_ == "mANN" or label_ == "mBNN" or label_ == "mCNN" or label_ == "mDNN" or label_ == "mENN" or label_ == "mFNN" or label_ == "mGNN":# or 
            #     plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle=':')
            #     # continue

            if label_ == "PBPF_RGBD_par_avg":
                label__labels.append('PBPF-RGBD')
            elif label_ == "FOUD":
                label__labels.append('FOUN')
            elif label_ == "Diff-DOPE":
                label__labels.append('Diff-DOPE')
            elif label_ == "PBPF_Opti_par_avg":
                label__labels.append('PBPF-Opti')
            elif label_ == "PBPF_RGB_par_avg":
                label__labels.append('PBPF-RGB')



        # Add labels and title
        plt.xlabel('Error Threshold (m)', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.title("AUC-"+ang_and_pos, fontsize=18)
        # plt.title("(Slow Push) (Fast Push)", fontsize=18)
        plt.legend(loc='lower right', labels=label__labels, title_fontsize=16)
        plt.grid(False)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.xlim([0, 0.1])
        plt.ylim([0, 1])
        plt.savefig(file_path+'AUC_'+ang_and_pos+'.svg', format='svg')
        plt.savefig(file_path+'AUC_'+ang_and_pos+'.png', format='png')


        areas = {}
 
        # Calculate the area under each curve using the composite trapezoidal rule
        for alg in result_df.index:
            y_values = [0] + result_df.loc[alg].values.tolist()
            area = simps(y_values, x_values)
            areas[alg] = area
        
        print("Areas under the curve for each algorithm:")
        print(areas)
        sorted_dict = dict(sorted(areas.items(),key=lambda item: item[1], reverse=True))
        for key,value in sorted_dict.items():
            print(f"{key}: {value}")

        # Normalize the areas so that the total area is 1
        total_area = sum(areas.values())
        normalized_areas = {alg: area / total_area for alg, area in areas.items()}
        
        print("normalized_areas:")
        print(normalized_areas)

