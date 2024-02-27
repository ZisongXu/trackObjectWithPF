import os
import sys
import yaml
from rosbag.bag import Bag

# info_dict = yaml.load(Bag('input.bag', 'r')._get_yaml_info())




object_name = sys.argv[1]
particle_number = int(sys.argv[2])
scene_name = sys.argv[3]
# run_alg_flag = sys.argv[4]
rosbag_num = sys.argv[4]



# info_dict = yaml.load(Bag(os.path.expanduser(f"~/rosbag/latest_rosbag/{object_name}_{scene_name}/{object_name}_{scene_name}_70_{rosbag_num}.bag"))._get_yaml_info(), Loader=yaml.FullLoader)
info_dict = yaml.load(Bag(os.path.expanduser(f"~/rosbag/depth_image_cracker_soup_barry{rosbag_num}.bag"))._get_yaml_info(), Loader=yaml.FullLoader)
# info_dict = yaml.load(Bag(os.path.expanduser(f"~/rosbag/pointcloud_cracker2.bag"))._get_yaml_info(), Loader=yaml.FullLoader)

dura = (info_dict['duration'] + 1) * 100

print(dura)
