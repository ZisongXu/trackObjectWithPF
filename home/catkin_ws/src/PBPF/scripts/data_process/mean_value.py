# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""
# gazebo_flag: false
# object_name_list:
# - soup
# - fish_can
# object_num: 1
# other_obj_num: 0
# oto_name_list:
# - base_of_cracker
# - fish_can
# particle_num: 140
# robot_num: 1
# run_alg_flag: PBPF
# task_flag: '3'
# update_style_flag: time
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
ang_and_pos_list = ["ADD", "ADDS"]

# normal_and_par_list = ["par"] # par/normal
# ang_and_pos_list = ["ADD"]



ang_and_pos_list_len = len(ang_and_pos_list)

error_thresholds = [0.001 * i for i in range(0, 101)]



columns_name = ['step', 'time', 'alg', 'obj', 'scene', 'particle_num', 'ray_type', 'obj_name', 'Errors']
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
        print(result_df)




        color_map = {
            "FOUD": "#FC8002",
            "DOPE": "#F0EEBB",
            "PBPF_RGBD_par_min": "#614099",
            "PBPF_RGB_par_min": "#EE4431",
            "PBPF_D_par_min": "#369F2D",
            "PBPF_RGBD_par_avg": "#614099",
            "PBPF_RGB_par_avg": "#EE4431",
            "PBPF_D_par_avg": "#369F2D",
            "Diff-DOPE": "#4995C6",
            "Diff-DOPE-Tracking": "#EDB11A",
        }



        error_thresholds = [0.001 * i for i in range(0, 101)]
        x_values = [0] + error_thresholds
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        
        for alg in result_df.index:
            y_values = [0] + result_df.loc[alg].values.tolist()
            if alg == "PBPF_RGBD_par_min" or alg == "PBPF_RGB_par_min" or alg == "PBPF_D_par_min":
                plt.plot(x_values, y_values, label=alg, color=color_map.get(alg,'#000000'), linestyle='--')
            else:
                plt.plot(x_values, y_values, label=alg, color=color_map.get(alg,'#000000'))
                
        
        # Add labels and title
        # plt.xlabel('Error Threshold (m)')
        # plt.ylabel('Accuracy')
        # plt.title(ang_and_pos+" AUC")
        # plt.legend(title='Algorithm')
        # plt.grid(False)
        # plt.xlim([0, 0.1])
        # plt.ylim([0, 1])
        # plt.savefig(file_path+'AUC.svg', format='svg')

        # # Show plot
        # plt.show()

        areas = {}
 
        # Calculate the area under each curve using the composite trapezoidal rule
        for alg in result_df.index:
            y_values = [0] + result_df.loc[alg].values.tolist()
            area = simps(y_values, x_values)
            areas[alg] = area
        
        print("Areas under the curve for each algorithm:")
        print(areas)

        # Normalize the areas so that the total area is 1
        total_area = sum(areas.values())
        normalized_areas = {alg: area / total_area for alg, area in areas.items()}
        
        print("normalized_areas:")
        print(normalized_areas)






# columns_name = ['step', 'time', 'alg', 'obj', 'scene', 'particle_num', 'ray_type', 'obj_name', 'Errors']
# for ang_and_pos in ang_and_pos_list:
#     ang_and_pos_AUC_list = []
#     for normal_and_par in normal_and_par_list:
#         panda_data_list = []
#         file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/results/0_error_all/"+normal_and_par+"/"+ang_and_pos+"/")
#         txt_file_count = len([file for file in os.listdir(file_path) if file.endswith('.csv')])
#         print("file_path:", file_path)
#         # for q in range(task_flag_list_len):
#         #     for w in range(object_name_list_len):
#         #         for e in range(ang_and_pos_list_len):
#         #             # based_on_time_70_scene1_time_Mustard_ADD
        
#         for csv_index in range(txt_file_count):
#             file_name = str(csv_index+1)+".csv"
#             # print("file_name:", file_name)
#             data = pd.read_csv(file_path+file_name, names=columns_name, header=None)
#             panda_data_list.append(data)

#         combined_data = pd.concat(panda_data_list)
        
#         ang_and_pos_AUC_list.append(combined_data)

#     combined_data = pd.concat(ang_and_pos_AUC_list)
#     error_means = combined_data.groupby('alg')['Errors'].mean()
#     average_errors_combined = combined_data.groupby(['obj_name', 'alg'])['Errors'].mean().reset_index()

#     print("error_means:", error_means)
#     print("average_errors_combined:", average_errors_combined)

#     result = {}
#     for alg, group in combined_data.groupby('alg'):
#         total_count = len(group)
#         proportions = {}
#         for threshold in error_thresholds:
#             count_below_threshold = len(group[group['Errors'] < threshold])
#             proportions[f'Errors < {threshold}'] = count_below_threshold / total_count
#         result[alg] = proportions
#     result_df = pd.DataFrame(result).transpose()
#     print(result_df)

#     color_map = {
#         "FOUD": "#FC8002",
#         "DOPE": "#F0EEBB",
#         "PBPF_RGBD_par_min": "#614099",
#         "PBPF_RGB_par_min": "#EE4431",
#         "PBPF_D_par_min": "#369F2D",
#         "PBPF_RGBD_par_avg": "#614099",
#         "PBPF_RGB_par_avg": "#EE4431",
#         "PBPF_D_par_avg": "#369F2D",
#         "Diff-DOPE": "#4995C6",
#         "Diff-DOPE-Tracking": "#EDB11A",
#     }

#     error_thresholds = [0.001 * i for i in range(0, 101)]
#     x_values = [0] + error_thresholds
    
#     # Plot the data
#     plt.figure(figsize=(10, 6))
#     alg_labels = []
#     for alg in result_df.index:
#         y_values = [0] + result_df.loc[alg].values.tolist()
#         if alg == "PBPF_RGBD_par_min" or alg == "PBPF_RGB_par_min" or alg == "PBPF_D_par_min":
#             plt.plot(x_values, y_values, label=alg, color=color_map.get(alg,'#000000'), linestyle='--')
#         else:
#             plt.plot(x_values, y_values, label=alg, color=color_map.get(alg,'#000000'))
#         if alg == "Diff-DOPE":
#             alg_labels.append("Diff-DOPE")
#         elif alg == "Diff-DOPE-Tracking":
#             alg_labels.append("Diff-DOPE (T)")
#         elif alg == "FOUD":
#             alg_labels.append("FOUD")
#         elif alg == "PBPF_RGBD_par_avg":
#             alg_labels.append("PBPF-RGBD")
#         elif alg == "PBPF_RGB_par_avg":
#             alg_labels.append("PBPF-RGB")
#         elif alg == "PBPF_D_par_avg":
#             alg_labels.append("PBPF-D")
#         elif alg == "PBPF_RGBD_par_min":
#             alg_labels.append("PBPF-RGBD (BP)")
#         elif alg == "PBPF_RGB_par_min":
#             alg_labels.append("PBPF-RGB (BP)")
#         elif alg == "PBPF_D_par_min":
#             alg_labels.append("PBPF-D (BP)")
    
#     # Add labels and title
#     plt.xlabel('Error Threshold (m)', fontsize=18)
#     plt.ylabel('Accuracy', fontsize=18)
#     plt.title(ang_and_pos+" AUC", fontsize=18)
#     plt.legend(title='Algorithm', labels=alg_labels, title_fontsize=16)
#     plt.grid(False)
    
#     plt.xticks(fontsize=17)
#     plt.yticks(fontsize=17)

#     plt.xlim([0, 0.1])
#     plt.ylim([0, 1])
#     plt.savefig(file_path+'AUC.svg', format='svg')

#     # Show plot
#     # plt.show()





















# columns_name = ['step', 'time', 'alg', 'obj', 'scene', 'particle_num', 'ray_type', 'obj_name', 'Errors']
# for ang_and_pos in ang_and_pos_list:
#     for normal_and_par in normal_and_par_list:
#         panda_data_list = []
#         file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/results/0_error_all/"+normal_and_par+"/"+ang_and_pos+"/")
#         txt_file_count = len([file for file in os.listdir(file_path) if file.endswith('.csv')])
#         print("file_path:", file_path)
#         # for q in range(task_flag_list_len):
#         #     for w in range(object_name_list_len):
#         #         for e in range(ang_and_pos_list_len):
#         #             # based_on_time_70_scene1_time_Mustard_ADD
        
#         for csv_index in range(txt_file_count):
#             file_name = str(csv_index+1)+".csv"
#             # print("file_name:", file_name)
#             data = pd.read_csv(file_path+file_name, names=columns_name, header=None)
#             panda_data_list.append(data)

#         combined_data = pd.concat(panda_data_list)
#         error_means = combined_data.groupby('alg')['Errors'].mean()
#         average_errors_combined = combined_data.groupby(['obj_name', 'alg'])['Errors'].mean().reset_index()

#         print("error_means:", error_means)
#         print("average_errors_combined:", average_errors_combined)

#         result = {}
#         for alg, group in combined_data.groupby('alg'):
#             total_count = len(group)
#             proportions = {}
#             for threshold in error_thresholds:
#                 count_below_threshold = len(group[group['Errors'] < threshold])
#                 proportions[f'Errors < {threshold}'] = count_below_threshold / total_count
#             result[alg] = proportions
#         result_df = pd.DataFrame(result).transpose()
#         print(result_df)

#         color_map = {
#             "FOUD": "#FC8002",
#             "DOPE": "#4995C6",
#             "PBPF_RGBD": "#614099",
#             "PBPF_RGB": "#EE4431",
#             "PBPF_D": "#369F2D",
#             "PBPF_RGBD_par_min": "#614099",
#             "PBPF_RGB_par_min": "#EE4431",
#             "PBPF_D_par_min": "#369F2D",
#             "Diff-DOPE": "#EDB11A",
#         }
#         color_map = {
#             "FOUD": "#FC8002",
#             "DOPE": "#F0EEBB",
#             "PBPF_RGBD_par_min": "#614099",
#             "PBPF_RGB_par_min": "#EE4431",
#             "PBPF_D_par_min": "#369F2D",
#             "PBPF_RGBD_par_avg": "#614099",
#             "PBPF_RGB_par_avg": "#EE4431",
#             "PBPF_D_par_avg": "#369F2D",
#             "Diff-DOPE": "#4995C6",
#             "Diff-DOPE-Tracking": "#EDB11A",
#         }
#         # color_map = {
#         #     "FOUD": "#FC8002",
#         #     "DOPE": "#F0EEBB",
#         #     "PBPF_RGBD_par_avg": "#614099",
#         #     "PBPF_RGB_par_avg": "#EE4431",
#         #     "PBPF_D_par_avg": "#369F2D",
#         #     "Diff-DOPE": "#4995C6",
#         #     "Diff-DOPE-Tracking": "#EDB11A",
#         # }


#         error_thresholds = [0.001 * i for i in range(0, 101)]
#         x_values = [0] + error_thresholds
        
#         # Plot the data
#         plt.figure(figsize=(10, 6))
        
#         for alg in result_df.index:
#             y_values = [0] + result_df.loc[alg].values.tolist()
#             if alg == "PBPF_RGBD_par_min" or alg == "PBPF_RGB_par_min" or alg == "PBPF_D_par_min":
#                 plt.plot(x_values, y_values, label=alg, color=color_map.get(alg,'#000000'), linestyle='--')
#             else:
#                 plt.plot(x_values, y_values, label=alg, color=color_map.get(alg,'#000000'))
                
        
#         # Add labels and title
#         plt.xlabel('Error Threshold (m)')
#         plt.ylabel('Accuracy')
#         plt.title(ang_and_pos+" AUC")
#         plt.legend(title='Algorithm')
#         plt.grid(False)
#         plt.xlim([0, 0.1])
#         plt.ylim([0, 1])
#         plt.savefig(file_path+'AUC.svg', format='svg')

#         # # Show plot
#         # plt.show()

#         areas = {}
 
#         # Calculate the area under each curve using the composite trapezoidal rule
#         for alg in result_df.index:
#             y_values = [0] + result_df.loc[alg].values.tolist()
#             area = simps(y_values, x_values)
#             areas[alg] = area
        
#         print("Areas under the curve for each algorithm:", areas)

#         # Normalize the areas so that the total area is 1
#         total_area = sum(areas.values())
#         normalized_areas = {alg: area / total_area for alg, area in areas.items()}
        
#         print("normalized_areas:", normalized_areas)

