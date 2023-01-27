# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""

import pandas as pd
import numpy as np
import os
import yaml

with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
    parameter_info = yaml.safe_load(file)
object_name_list = parameter_info['object_name_list']
pos_flag = False
ang_flag = True
object_flag = object_name_list[0]
task_flag = "1" #1/2/3/4
def stackcsv(content_folder):
    with open(file_name, "w") as fdout:
        entries = os.listdir(content_folder)
        for i in entries:
            csv_path = os.path.join(content_folder, i)
            with open(csv_path) as fdin:
                while True:
                    chunk = fdin.read()
                    if len(chunk) == 0: break
                    fdout.write(chunk)
if pos_flag == True:
    style = "pos"
if ang_flag == True:
    style = "ang"



file_name = object_flag+"_time_scene"+task_flag+"_"+style+".csv"
content_folder = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/error_file/cracker_scene1_70/inter_data_"+style+"/")


stackcsv(content_folder)

# os.chdir( )
# file_chdir = os.getcwd()

# filecsv_list = []
# for root, dirs, files in os.walk( file_chdir ):
#     for file in files:
#         if os.path.splitext( file )[1] == '.csv':
#             filecsv_list.append( file )

# data = pd.DataFrame()
# for csv in filecsv_list:
#     # data = data.append( pd.read_csv( csv, header=0, sep=None, encoding='gb18030' ) )
#     data = data.append( pd.read_csv( csv ) )

# data.to_csv( 'ALL.csv', index=0 )
