import os
import sys
import yaml

object_name = sys.argv[1]
particle_number = int(sys.argv[2])
scene_name = sys.argv[3]
run_alg_flag = sys.argv[4]
version = sys.argv[5]

with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
        parameter_info = yaml.safe_load(file)
        parameter_info['particle_num'] = particle_number
        parameter_info['object_name_list'][0] = object_name
        parameter_info['task_flag'] = scene_name[-1]
        parameter_info['run_alg_flag'] = run_alg_flag
        parameter_info['version'] = version

with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'w') as file:
        yaml.dump(parameter_info, file, default_flow_style=False)
