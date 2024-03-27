#!/bin/bash

declare -a objectNames=("cracker" "soup")
# declare -a objectNames=("cracker" "gelatin" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a sceneNames=("scene2")

declare -a particleNumbers=(70)
# declare -a objectNames=("cracker")
# declare -a sceneNames=("scene3")
declare -a runAlgFlags=("obse" "PBPFV")
# declare -a Ang_and_Pos=("ang" "pos")
declare -a Ang_and_Pos=("ADD")
declare -a update_style_flag=("time") # "time" "pose"
# declare -a runVersions=("depth_img" "multiray")
declare -a runVersions=("PBPF_RGBD" "PBPF_RGB" "PBPF_D")
# for runAlgFlag in "${runAlgFlags[@]}"
# do
# 	for particleNumber in "${particleNumbers[@]}"
# 	do
# 		for objectName in "${objectNames[@]}"
# 		do
			
# 			for sceneName in "${sceneNames[@]}"
# 			do
# 				if [[ "$objectName" == "soup" ]]; then
# 					if [[ "$sceneName" == "scene4" ]]; then
# 						continue
# 					fi
# 				fi
				
# 				python3 update_yaml_file_automated.py "${objectName}" "${particleNumber}" "${sceneName}" "${runAlgFlag}"
				
# 				for rosbag in {1..10}
# 				do

# particle_num = sys.argv[1]
# object_name = sys.argv[2]
# task_flag = sys.argv[3] # "scene1"
# rosbag_flag = sys.argv[4]
# repeat_time = sys.argv[5]
# run_alg_flag = sys.argv[6] # PBPF
# ang_and_pos = sys.argv[7] # pos/ang

for ang_and_pos in "${Ang_and_Pos[@]}"
do
	for objectName in "${objectNames[@]}"
	do
		for sceneName in "${sceneNames[@]}"
		do
			for particleNumber in "${particleNumbers[@]}"
			do
				for runAlgFlag in "${runAlgFlags[@]}"
				do
					if [[ "$objectName" == "soup" ]]; then
						if [[ "$sceneName" == "scene4" ]]; then
							continue
						fi
					fi
					# for rosbag in {1..10}
					for ((rosbag=1;rosbag<=1;rosbag++)); 
					do
						# for repeat in {1..10}
						for ((repeat=0;repeat<=9;repeat++));
						do
							for runVersion in "${runVersions[@]}"
							do
								python3 data_processing.py "${particleNumber}" "${objectName}" "${sceneName}" "${rosbag}" "${repeat}" "${runAlgFlag}" "${ang_and_pos}" "${runVersion}" &
								DATA_PRO_PID=$!

								sleep 4
							done
						done
					done
				done
			done
		done
	done
done
# 				done

# 			done
# 		done
# 	done
# done





# for particleNumber in "${particleNumbers[@]}"
# do
# 	for objectName in "${objectNames[@]}"
# 	do
# 		for sceneName in "${sceneNames[@]}"
# 		do
# 			for update_style in "${update_style_flag[@]}"
# 			do
# 				for ang_and_pos in "${Ang_and_Pos[@]}"
# 				do
# 					# python3 inter_data.py "${ang_and_pos}" &
# 					# INTER_DATA_PID=$!
# 					# sleep 5

# 					python3 plotsns_base_on_time.py "${particleNumber}" "${objectName}" "${sceneName}" "${update_style}" "${ang_and_pos}" &
# 					# PLOT_PID=$!
# 					# sleep 100
# 				done
# 			done
# 		done
# 	done
# done
