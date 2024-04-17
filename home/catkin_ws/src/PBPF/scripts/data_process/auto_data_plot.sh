#!/bin/bash

declare -a objectNames=("cracker" "soup")
# declare -a objectNames=("cracker" "gelatin" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a sceneNames=("scene1")

declare -a particleNumbers=(70)
# declare -a objectNames=("cracker")
# declare -a sceneNames=("scene3")
declare -a runAlgFlags=("PBPFV" "obse")

declare -a runVersions=("PBPF_RGBD" "PBPF_RGB" "PBPF_D")
# declare -a Ang_and_Pos=("pos" "ang")
declare -a Ang_and_Pos=("ADD")
declare -a update_style_flag=("time") # "time" "pose"


for particleNumber in "${particleNumbers[@]}"
do
	for objectName in "${objectNames[@]}"
	do
		for sceneName in "${sceneNames[@]}"
		do
			for update_style in "${update_style_flag[@]}"
			do
				for ang_and_pos in "${Ang_and_Pos[@]}"
				do
					for ((rosbag=1;rosbag<=1;rosbag++)); 
					do
						# python3 inter_data.py "${ang_and_pos}" &
						# INTER_DATA_PID=$!
						# sleep 5

						python3 plotsns_base_on_time.py "${particleNumber}" "${objectName}" "${sceneName}" "${rosbag}" "${update_style}" "${ang_and_pos}" &
						# PLOT_PID=$!
						# sleep 100
					done
				done
			done
		done
	done
done
