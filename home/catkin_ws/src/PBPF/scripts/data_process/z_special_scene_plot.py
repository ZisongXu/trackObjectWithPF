# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""



import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# Define the column names
columns_name = ['step','time','alg','obj','scene','particle_num','ray_type','obj_name','error']
 
# Load the CSV files with the defined column names
file_paths = [
    'based_on_time_40_scene2_rosbag1_time_SaladDressing_ADD_par_avg.csv',
    'based_on_time_40_scene2_rosbag1_time_Mustard_ADD_par_avg.csv',
    'based_on_time_40_scene2_rosbag1_time_Mayo_ADD_par_avg.csv'
]
# file_paths = [
#     'based_on_time_40_scene2_rosbag1_time_SaladDressing_ADD_par_min.csv',
#     'based_on_time_40_scene2_rosbag1_time_Mustard_ADD_par_min.csv',
#     'based_on_time_40_scene2_rosbag1_time_Mayo_ADD_par_min.csv'
# ]


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
#         "FOUD": "#FC8002",
#         "DOPE": "#F0EEBB",
#         "PBPF_RGBD_par_min": "#614099",
#         "PBPF_RGB_par_min": "#EE4431",
#         "PBPF_D_par_min": "#369F2D",
#         "Diff-DOPE": "#4995C6",
#         "Diff-DOPE-Tracking": "#EDB11A",
#     }




dashes_map = {     
    'SaladDressing': '',         # 实线
    'Mustard': (4, 2),     # 短划线
    'Mayo': (1, 2),     # 点线
    'obj4': (1, 2)      # 点划线 
    }


x_range_max = 115 # 28, 129, 265
x_range_unit = 12 # 2, 6, 25, 125
y_range_max = 0.5 # 0.5
y_range_unit = 0.05 # 0.04
x_xlim = 115 # 28
y_ylim = 0.5 # 0.5

dataframes = [pd.read_csv(file_path, names=columns_name) for file_path in file_paths]
 
# Display the first few rows of each dataframe to verify
# print(dataframes[0].head())
# print(dataframes[1].head())
# print(dataframes[2].head())

combined_df = pd.concat(dataframes, ignore_index=True)
# print(combined_df.head())
print(len(combined_df))

# Filter the combined dataframe for rows where 'alg' is 'PBPF_RGBD_par_avg' or 'FOUD'
filtered_df = combined_df[combined_df['alg'].isin(['PBPF_RGBD_par_avg', 'FOUD'])]
# filtered_df = combined_df[combined_df['alg'].isin(['PBPF_RGBD_par_min', 'FOUD'])]
 
# Display the first few rows of the filtered dataframe to verify
# print(filtered_df.head())
print(len(filtered_df))
# print(filtered_df2.head())


filtered_df.columns=["index","time","alg","obj","scene","particle_num","ray_type","obj_name","ADD Error (m)"]
figure_ADD = sns.lineplot(data=filtered_df, x="time", y="ADD Error (m)", hue='alg', style='obj', dashes=dashes_map, errorbar=('ci', 95), legend=True, linewidth=0.5, palette=color_map)
figure_ADD.set(xlabel = None, ylabel = None)
x = range(0, x_range_max, x_range_unit)
y = np.arange(0, y_range_max, y_range_unit)
plt.xticks(x)
plt.yticks(y)
plt.tick_params(labelsize=15)
plt.xlim(0, x_xlim)
plt.ylim(0, y_ylim)




# 获取当前图例对象
handles, labels = plt.gca().get_legend_handles_labels()

new_alg_labels = {     
    'PBPF_RGBD_par_avg': 'PBPF-RGBD',
    'FOUD': 'FOUD'
    } 
# new_alg_labels = {     
#     'PBPF_RGBD_par_min': 'PBPF-RGBD (BP)',
#     'FOUD': 'FOUD'
#     } 

# 自定义图例标签
new_labels = []
for label in labels:
    if 'alg' in label:
        new_labels.append(f'Algorithm')  # 自定义 alg 名称
    elif 'obj' in label:
        new_labels.append(f'Object Name')    # 自定义 obj 名称
    elif label in new_alg_labels:
        new_labels.append(new_alg_labels[label])  # 替换 alg 名称
    else:
        new_labels.append(label)


# new_labels = []
# for label in labels:
#     if label in new_alg_labels:
#         new_labels.append(new_alg_labels[label])  # 替换 alg 名称
#     elif label in new_obj_labels:
#         new_labels.append(new_obj_labels[label])  # 替换 obj 名称
#     else:
#         new_labels.append(label)  # 保留其他名称不变


# 设置新的图例
plt.legend(handles, new_labels)


plt.title("ADD errors (m) vs Time (s)", fontsize=16)
svg_fig_ADD = figure_ADD.get_figure()
svg_fig_ADD.savefig("z_par_avg.svg",format="svg")
# svg_fig_ADD.savefig("z_par_min.svg",format="svg")
print("finished")

# color_map = {
#     'PBPF_RGBD_par_avg': '#614099',
#     'FOUD': '#FC8002'
# }

# objects = filtered_df['obj'].unique() 
# algs = filtered_df['alg'].unique()
 
# # Create a plot for each combination of 'obj' and 'alg'
# fig, ax = plt.subplots(figsize=(12, 8))
 
# # Plot each line for the combinations of 'obj' and 'alg' with specified colors
# for obj in objects:
#     for alg in algs:
#         subset = filtered_df[(filtered_df['obj'] == obj) & (filtered_df['alg'] == alg)]
#         if not subset.empty:
#             ax.plot(subset['time'], subset['error'], label=f"{obj}-{alg}", color=color_map[alg])
 
# # Set labels and title
# ax.set_xlabel('Time')
# ax.set_ylabel('Error')
# ax.set_title('Error over Time by Object and Algorithm')
# ax.legend(loc='best')
 
# # Display the plot
# plt.show()