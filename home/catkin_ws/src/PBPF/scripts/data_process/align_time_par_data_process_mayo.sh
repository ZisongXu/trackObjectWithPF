#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a objectNames=("cracker" "Ketchup")
# declare -a objectNames=("soup" "Parmesan")
# declare -a objectNames=("SaladDressing" "Mustard")
# declare -a objectNames=("cracker" "gelatin" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
# declare -a sceneNames=("scene1")

# declare -a objectNames=("Mayo" "Milk")
# declare -a sceneNames=("scene1" "scene2")

# declare -a objectNames=("cracker" "soup" "Parmesan")
# declare -a objectNames=("cracker" "Mayo" "Milk")
# declare -a objectNames=("cracker")
# declare -a objectNames=("Ketchup" "Mayo" "Milk" "SaladDressing" "soup" "Parmesan" "Mustard")
declare -a objectNames=("Mayo")
declare -a sceneNames=("scene2")


declare -a particleNumbers=(70)
# declare -a objectNames=("cracker")
# declare -a sceneNames=("scene3")
declare -a runAlgFlags=("PBPF")
# declare -a Ang_and_Pos=("ang" "pos")
declare -a Ang_and_Pos=("ADD")
declare -a update_style_flag=("time") # "time" "pose"
# declare -a runVersions=("depth_img" "multiray")
declare -a runVersions=("PBPF_RGBD" "PBPF_RGB" "PBPF_D")
# declare -a runVersions=("PBPF_RGBD")

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
						for ((repeat=0;repeat<=4;repeat++));
						do
							for runVersion in "${runVersions[@]}"
							do
								for ((par_index=0;par_index<${particleNumber};par_index++)); 
								do
									python3 align_time_par_data_process_mayo.py "${particleNumber}" "${objectName}" "${sceneName}" "${rosbag}" "${repeat}" "${runAlgFlag}" "${ang_and_pos}" "${runVersion}" "${par_index}"&
									DATA_PRO_PID=$!

									sleep 2
								done
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
