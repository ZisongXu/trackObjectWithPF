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
# mass_and_friction_list = ["mass", "friction"] # par/normal
mass_and_friction_list = ["friction"] # par/normal/friction/mass/motion_noise/MassFriction_Var/
ang_and_pos_list = ["ADD", "ADDS"]
# ang_and_pos_list = ["ADDS"]

# mass_and_friction_list = ["mass"] # par/normal
# normal_and_par_list = ["par"] # par/normal
# ang_and_pos_list = ["ADD"]



ang_and_pos_list_len = len(ang_and_pos_list)

error_thresholds = [0.001 * i for i in range(0, 101)]



# columns_name = ['step', 'time', 'alg', 'obj', 'scene', 'particle_num', 'ray_type', 'obj_name', 'Errors']
columns_name = ['step', 'time', 'alg', 'obj', 'scene', 'particle_num', 'ray_type', 'obj_name', 'mass', 'friction', 'Errors']
for ang_and_pos in ang_and_pos_list:
    # for normal_and_par in normal_and_par_list:
    for mass_and_friction in mass_and_friction_list:
        panda_data_list = []
        # file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/results/0_error_all/"+normal_and_par+"/"+ang_and_pos+"/")
        file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/results/0_error_all/"+mass_and_friction+"/"+ang_and_pos+"/")
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
        if mass_and_friction == "motion_noise":
            error_means = combined_data.groupby('mass')['Errors'].mean()
            average_errors_combined = combined_data.groupby(['obj_name', 'mass'])['Errors'].mean().reset_index()

            print("error_means:", error_means)
            print("average_errors_combined:", average_errors_combined)

            result = {}
            for label_, group in combined_data.groupby('mass'):
                total_count = len(group)
                proportions = {}
                for threshold in error_thresholds:
                    count_below_threshold = len(group[group['Errors'] < threshold])
                    proportions[f'Errors < {threshold}'] = count_below_threshold / total_count
                result[label_] = proportions
            result_df = pd.DataFrame(result).transpose()
            print(result_df)
        
        elif mass_and_friction == "MassFriction_Var":
            error_means = combined_data.groupby('mass')['Errors'].mean()
            average_errors_combined = combined_data.groupby(['obj_name', 'mass'])['Errors'].mean().reset_index()

            print("error_means:", error_means)
            print("average_errors_combined:", average_errors_combined)

            result = {}
            for label_, group in combined_data.groupby('mass'):
                total_count = len(group)
                proportions = {}
                for threshold in error_thresholds:
                    count_below_threshold = len(group[group['Errors'] < threshold])
                    proportions[f'Errors < {threshold}'] = count_below_threshold / total_count
                result[label_] = proportions
            result_df = pd.DataFrame(result).transpose()
            print(result_df)

        else:
            error_means = combined_data.groupby(mass_and_friction)['Errors'].mean()
            average_errors_combined = combined_data.groupby(['obj_name', mass_and_friction])['Errors'].mean().reset_index()

            print("error_means:", error_means)
            print("average_errors_combined:", average_errors_combined)

            result = {}
            for label_, group in combined_data.groupby(mass_and_friction):
                total_count = len(group)
                proportions = {}
                for threshold in error_thresholds:
                    count_below_threshold = len(group[group['Errors'] < threshold])
                    proportions[f'Errors < {threshold}'] = count_below_threshold / total_count
                result[label_] = proportions
            result_df = pd.DataFrame(result).transpose()
            print(result_df)

        if mass_and_friction == "mass":
            color_map = {
                "mA": "#369F2D",
                "mAN": "#369F2D",
                "mANN": "#369F2D",
                "mB": "#614099",
                "mBN": "#614099",
                "mBNN": "#614099",
                "mC": "#EE4431",
                "mCN": "#EE4431",
                "mCNN": "#EE4431",
                "mD": "#FC8002",
                "mDN": "#FC8002",
                "mDNN": "#FC8002",
                "mE": "#F0EEBB",
                "mEN": "#F0EEBB",
                "mENN": "#F0EEBB",
                "mF": "#EDB11A",
                "mFN": "#EDB11A",
                "mFNN": "#EDB11A",
                "mG": "#4995C6",
                "mGN": "#4995C6",
                "mGNN": "#4995C6",
            }
        elif mass_and_friction == "friction":
            color_map = {
                "fA": "#369F2D",
                "fAN": "#369F2D",
                "fANN": "#369F2D",
                "fB": "#614099",
                "fBN": "#614099",
                "fBNN": "#614099",
                "fC": "#EE4431",
                "fCN": "#EE4431",
                "fCNN": "#EE4431",
                "fD": "#FC8002",
                "fDN": "#FC8002",
                "fDNN": "#FC8002",
                "fE": "#F0EEBB",
                "fEN": "#F0EEBB",
                "fENN": "#F0EEBB",
                "fF": "#EDB11A",
                "fFN": "#EDB11A",
                "fFNN": "#EDB11A",
                "fG": "#4995C6",
                "fGN": "#4995C6",
                "fGNN": "#4995C6",
                "fH": "#000000",
                "fHN": "#000000",
                "fHNN": "#000000",
            }
        elif mass_and_friction == "motion_noise":
            color_map = {
                "mA": "#369F2D",
                "mAN": "#369F2D",
                "mANN": "#369F2D",
                "mB": "#614099",
                "mBN": "#614099",
                "mBNN": "#614099",
                "mC": "#EE4431",
                "mCN": "#EE4431",
                "mCNN": "#EE4431",
                "mD": "#FC8002",
                "mDN": "#FC8002",
                "mDNN": "#FC8002",
                "mE": "#F0EEBB",
                "mEN": "#F0EEBB",
                "mENN": "#F0EEBB",
                "mF": "#EDB11A",
                "mFN": "#EDB11A",
                "mFNN": "#EDB11A",
                "mG": "#4995C6",
                "mGN": "#4995C6",
                "mGNN": "#4995C6",
            }
        elif mass_and_friction == "MassFriction_Var":
            color_map = {
                "mA": "#369F2D",
                "mAN": "#369F2D",
                "mANN": "#369F2D",
                "mB": "#614099",
                "mBN": "#614099",
                "mBNN": "#614099",
                "mC": "#EE4431",
                "mCN": "#EE4431",
                "mCNN": "#EE4431",
                "mD": "#FC8002",
                "mDN": "#FC8002",
                "mDNN": "#FC8002",
                "mE": "#F0EEBB",
                "mEN": "#F0EEBB",
                "mENN": "#F0EEBB",
                "mF": "#EDB11A",
                "mFN": "#EDB11A",
                "mFNN": "#EDB11A",
                "mG": "#4995C6",
                "mGN": "#4995C6",
                "mGNN": "#4995C6",
            }

        error_thresholds = [0.001 * i for i in range(0, 101)]
        x_values = [0] + error_thresholds
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        label__labels = []
        for label_ in result_df.index:
            y_values = [0] + result_df.loc[label_].values.tolist()
            
            if mass_and_friction == "mass":
                
                if label_ == "mA" or label_ == "mB" or label_ == "mC" or label_ == "mD" or label_ == "mE" or label_ == "mF" or label_ == "mG":# or 
                    # plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'))
                    continue
                elif label_ == "mAN" or label_ == "mBN" or label_ == "mCN" or label_ == "mDN" or label_ == "mEN" or label_ == "mFN" or label_ == "mGN" :# or 
                    # plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle='--')
                    continue
                elif label_ == "mANN" or label_ == "mBNN" or label_ == "mCNN" or label_ == "mDNN" or label_ == "mENN" or label_ == "mFNN" or label_ == "mGNN":# or 
                    # plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle=':')
                    plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'))
                    # continue

                if label_ == "mA":
                    label__labels.append('mA = 0.01')
                elif  label_ == "mAN":
                    label__labels.append('mAN = 0.01')
                elif  label_ == "mANN":
                    label__labels.append('mANN = 0')
                elif label_ == "mB":
                    label__labels.append('mB = True Mass')
                elif label_ == "mBN":
                    label__labels.append('mBN = True Mass')
                elif label_ == "mBNN":
                    label__labels.append('mBNN = 0.005/0.05')
                elif label_ == "mC":
                    label__labels.append('mC = 0.1')
                elif label_ == "mCN":
                    label__labels.append('mCN = 0.1')
                elif label_ == "mCNN":
                    label__labels.append('mCNN = 0.1')
                elif label_ == "mD":
                    label__labels.append('mD = 0.5')
                elif label_ == "mDN":
                    label__labels.append('mDN = 0.5')
                elif label_ == "mDNN":
                    label__labels.append('mDNN = 0.5')
                elif label_ == "mE":
                    label__labels.append('mE = 1.0')
                elif label_ == "mEN":
                    label__labels.append('mEN = 1.0')
                elif label_ == "mENN":
                    label__labels.append('mENN = 1.0')
                elif label_ == "mF":
                    label__labels.append('mF = 5.0')
                elif label_ == "mFN":
                    label__labels.append('mFN = 5.0')
                elif label_ == "mFNN":
                    label__labels.append('mFNN = 5.0')
                elif label_ == "mG":
                    label__labels.append('mG = 10.')
                elif label_ == "mGN":
                    label__labels.append('mGN = 10.')
                elif label_ == "mGNN":
                    label__labels.append('mGNN = 10.')



            elif mass_and_friction == "friction":
                # if label_ == "fE" or label_ == "fF" or label_ == "fG" or label_ == "fH" or label_ == "fA" or label_ == "fB" or label_ == "fC" or label_ == "fD":# or 
                #     plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'))
                #     # continue
                # elif label_ == "fEN" or label_ == "fFN" or label_ == "fGN" or label_ == "fHN" or label_ == "fAN" or label_ == "fBN" or label_ == "fCN" or label_ == "fDN":# or 
                #     plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle='--')
                #     # continue
                # elif label_ == "fENN" or label_ == "fFNN" or label_ == "fGNN" or label_ == "fHNN" or label_ == "fANN" or label_ == "fBNN" or label_ == "fCNN" or label_ == "fDNN":# or 
                #     plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle=':')
                #     # continue

                if label_ == "fA" or label_ == "fB" or label_ == "fC" or label_ == "fD" or label_ == "fE" or label_ == "fF" or label_ == "fG":# or 
                    plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'))
                    # continue
                elif label_ == "fAN" or label_ == "fBN" or label_ == "fCN" or label_ == "fDN" or label_ == "fEN" or label_ == "fFN" or label_ == "fGN" :# or 
                    plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle='--')
                    # continue
                elif label_ == "fANN" or label_ == "fBNN" or label_ == "fCNN" or label_ == "fDNN" or label_ == "fENN" or label_ == "fFNN" or label_ == "fGNN":# or 
                    plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle=':')
                    # continue



                if label_ == "fA":
                    label__labels.append('fA = 0.01')
                elif label_ == "fAN":
                    label__labels.append('fAN = 0.01')
                elif label_ == "fANN":
                    label__labels.append('fANN = 0.01')
                elif label_ == "fB":
                    label__labels.append('fB = 0.10')
                elif label_ == "fBN":
                    label__labels.append('fBN = 0.10')
                elif label_ == "fBNN":
                    label__labels.append('fBNN = 0.10')
                elif label_ == "fC":
                    label__labels.append('fC = 0.25')
                elif label_ == "fCN":
                    label__labels.append('fCN = 0.25')
                elif label_ == "fCNN":
                    label__labels.append('fCNN = 0.25')
                elif label_ == "fD":
                    label__labels.append('fD = 0.38')
                elif label_ == "fDN":
                    label__labels.append('fDN = 0.38')
                elif label_ == "fDNN":
                    label__labels.append('fDNN = 0.38')
                elif label_ == "fE":
                    label__labels.append('fE = 0.50')
                elif label_ == "fEN":
                    label__labels.append('fEN = 0.50')
                elif label_ == "fENN":
                    label__labels.append('fENN = 0.50')
                elif label_ == "fF":
                    label__labels.append('fF = 0.75')
                elif label_ == "fFN":
                    label__labels.append('fFN = 0.75')
                elif label_ == "fFNN":
                    label__labels.append('fFNN = 0.75')
                elif label_ == "fG":
                    label__labels.append('fG = 1.00')
                elif label_ == "fGN":
                    label__labels.append('fGN = 1.00')
                elif label_ == "fGNN":
                    label__labels.append('fGNN = 1.00')


                # if label_ == "fA":
                #     label__labels.append('fA = 0.01')
                # elif label_ == "fAN":
                #     label__labels.append('fAN = 0.01')
                # elif label_ == "fANN":
                #     label__labels.append('fANN = 0.01')
                # elif label_ == "fB":
                #     label__labels.append('fB = 0.10')
                # elif label_ == "fBN":
                #     label__labels.append('fBN = 0.10')
                # elif label_ == "fBNN":
                #     label__labels.append('fBNN = 0.10')
                # elif label_ == "fC":
                #     label__labels.append('fC = 0.25')
                # elif label_ == "fCN":
                #     label__labels.append('fCN = 0.25')
                # elif label_ == "fCNN":
                #     label__labels.append('fCNN = 0.25')
                # elif label_ == "fD":
                #     label__labels.append('fD = 0.50')
                # elif label_ == "fDN":
                #     label__labels.append('fDN = 0.50')
                # elif label_ == "fDNN":
                #     label__labels.append('fDNN = 0.50')
                # elif label_ == "fE":
                #     label__labels.append('fE = 0.75')
                # elif label_ == "fEN":
                #     label__labels.append('fEN = 0.75')
                # elif label_ == "fENN":
                #     label__labels.append('fENN = 0.75')
                # elif label_ == "fF":
                #     label__labels.append('fF = 1.00')
                # elif label_ == "fFN":
                #     label__labels.append('fFN = 1.00')
                # elif label_ == "fFNN":
                #     label__labels.append('fFNN = 1.00')
                # elif label_ == "fG":
                #     label__labels.append('fF = 1.00')
                # elif label_ == "fGN":
                #     label__labels.append('fFN = 1.00')
                # elif label_ == "fGNN":
                #     label__labels.append('fFNN = 1.00')


                # elif label_ == "fG":
                #     label__labels.append('fG = 0.30')
                # elif label_ == "fGN":
                #     label__labels.append('fGN = 0.30')
                # elif label_ == "fGNN":
                #     label__labels.append('fGNN = 0.30')
                # elif label_ == "fH":
                #     label__labels.append('fH = 0.40')
                # elif label_ == "fHN":
                #     label__labels.append('fHN = 0.40')
                # elif label_ == "fHNN":
                #     label__labels.append('fHNN = 0.40')
            elif mass_and_friction == "motion_noise":
                
                if label_ == "mA" or label_ == "mB" or label_ == "mC" or label_ == "mD" or label_ == "mE" or label_ == "mF" or label_ == "mG":# or 
                    # plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'))
                    continue
                elif label_ == "mAN" or label_ == "mBN" or label_ == "mCN" or label_ == "mDN" or label_ == "mEN" or label_ == "mFN" or label_ == "mGN" :# or 
                    # plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle='--')
                    continue
                elif label_ == "mANN" or label_ == "mBNN" or label_ == "mCNN" or label_ == "mDNN" or label_ == "mENN" or label_ == "mFNN" or label_ == "mGNN":# or 
                    # plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle=':')
                    plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'))
                    # continue

                if label_ == "mA":
                    label__labels.append('mA = 0.01')
                elif  label_ == "mAN":
                    label__labels.append('mAN = 0.01')
                elif  label_ == "mANN":
                    label__labels.append('sigmaF = 0.000/0.00')
                elif label_ == "mB":
                    label__labels.append('mB = True Mass')
                elif label_ == "mBN":
                    label__labels.append('mBN = True Mass')
                elif label_ == "mBNN":
                    label__labels.append('sigmaF = 0.005/0.05')
                elif label_ == "mC":
                    label__labels.append('mC = 0.1')
                elif label_ == "mCN":
                    label__labels.append('mCN = 0.1')
                elif label_ == "mCNN":
                    label__labels.append('sigmaF = 0.010/0.10')
                elif label_ == "mD":
                    label__labels.append('mD = 0.5')
                elif label_ == "mDN":
                    label__labels.append('mDN = 0.5')
                elif label_ == "mDNN":
                    label__labels.append('sigmaF = 0.020/0.20')
                elif label_ == "mE":
                    label__labels.append('mE = 1.0')
                elif label_ == "mEN":
                    label__labels.append('mEN = 1.0')
                elif label_ == "mENN":
                    label__labels.append('sigmaF = 0.050/0.50')
                elif label_ == "mF":
                    label__labels.append('mF = 5.0')
                elif label_ == "mFN":
                    label__labels.append('mFN = 5.0')
                elif label_ == "mFNN":
                    label__labels.append('mFNN = 5.0')
                elif label_ == "mG":
                    label__labels.append('mG = 10.')
                elif label_ == "mGN":
                    label__labels.append('mGN = 10.')
                elif label_ == "mGNN":
                    label__labels.append('mGNN = 10.')
        
            elif mass_and_friction == "MassFriction_Var":
                
                if label_ == "mA" or label_ == "mB" or label_ == "mC" or label_ == "mD" or label_ == "mE" or label_ == "mF" or label_ == "mG":# or 
                    # plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'))
                    continue
                elif label_ == "mAN" or label_ == "mBN" or label_ == "mCN" or label_ == "mDN" or label_ == "mEN" or label_ == "mFN" or label_ == "mGN" :# or 
                    # plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle='--')
                    continue
                elif label_ == "mANN" or label_ == "mBNN" or label_ == "mCNN" or label_ == "mDNN" or label_ == "mENN" or label_ == "mFNN" or label_ == "mGNN":# or 
                    # plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'), linestyle=':')
                    plt.plot(x_values, y_values, label=label_, color=color_map.get(label_,'#000000'))
                    # continue

            
                if label_ == "mA":
                    label__labels.append('mA = 0.01')
                elif  label_ == "mAN":
                    label__labels.append('mAN = 0.01')
                elif  label_ == "mANN":
                    label__labels.append('TSigma / 100.')
                elif label_ == "mB":
                    label__labels.append('mB = True Mass')
                elif label_ == "mBN":
                    label__labels.append('mBN = True Mass')
                elif label_ == "mBNN":
                    label__labels.append('TSigma')
                elif label_ == "mC":
                    label__labels.append('mC = 0.1')
                elif label_ == "mCN":
                    label__labels.append('mCN = 0.1')
                elif label_ == "mCNN":
                    label__labels.append('TSigma / 10.0')
                elif label_ == "mD":
                    label__labels.append('mD = 0.5')
                elif label_ == "mDN":
                    label__labels.append('mDN = 0.5')
                elif label_ == "mDNN":
                    label__labels.append('TSigma * 2.00')
                elif label_ == "mE":
                    label__labels.append('mE = 1.0')
                elif label_ == "mEN":
                    label__labels.append('mEN = 1.0')
                elif label_ == "mENN":
                    label__labels.append('TSigma * 5.00')
                elif label_ == "mF":
                    label__labels.append('mF = 5.0')
                elif label_ == "mFN":
                    label__labels.append('mFN = 5.0')
                elif label_ == "mFNN":
                    label__labels.append('TSigma * 10.0')
                elif label_ == "mG":
                    label__labels.append('mG = 10.')
                elif label_ == "mGN":
                    label__labels.append('mGN = 10.')
                elif label_ == "mGNN":
                    label__labels.append('TSigma * 100.')



        # Add labels and title
        plt.xlabel('Error Threshold (m)', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.title("AUC-"+ang_and_pos, fontsize=18)
        plt.legend(loc='lower right', labels=label__labels, title_fontsize=16)
        plt.grid(False)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.xlim([0, 0.1])
        plt.ylim([0, 1])
        plt.savefig(file_path+'AUC_'+ang_and_pos+'_'+mass_and_friction+'.svg', format='svg')
        plt.savefig(file_path+'AUC_'+ang_and_pos+'_'+mass_and_friction+'.png', format='png')

        # # Show plot
        # plt.show()

        areas = {}
 
        # Calculate the area under each curve using the composite trapezoidal rule
        for label_ in result_df.index:
            y_values = [0] + result_df.loc[label_].values.tolist()
            area = simps(y_values, x_values)
            areas[label_] = area
        
        print("Areas under the curve for each algorithm:")
        print(areas)
        sorted_dict = dict(sorted(areas.items(),key=lambda item: item[1], reverse=True))
        for key,value in sorted_dict.items():
            print(f"{key}: {value}")

        # Normalize the areas so that the total area is 1
        total_area = sum(areas.values())
        normalized_areas = {label_: area / total_area for label_, area in areas.items()}
        
        print("normalized_areas:")
        print(normalized_areas)



