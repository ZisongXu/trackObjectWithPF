#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a objectNames=("cracker" "Ketchup")
# declare -a objectNames=("SaladDressing" "cracker")
declare -a objectNames=("cracker" "Milk" "Ketchup")
# declare -a objectNames=("Ketchup" "Mayo" "Milk" "SaladDressing" "soup" "Parmesan" "Mustard" "cracker")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a sceneNames=("scene2")

declare -a particleNumbers=(40)
# declare -a objectNames=("cracker")
# declare -a sceneNames=("scene3")
declare -a runAlgFlags=("DiffDOPE" "DiffDOPET")
# declare -a runAlgFlags=("FOUD")
# declare -a Ang_and_Pos=("ang" "pos")
declare -a Ang_and_Pos=("ADD")
declare -a update_style_flag=("time") # "time" "pose"
# declare -a runVersions=("depth_img" "multiray")
declare -a runVersions=("PBPF_RGBD")

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
						for ((repeat=0;repeat<=0;repeat++));
						do
							for runVersion in "${runVersions[@]}"
							do
								python3 align_time_diffDOPE_data_process.py "${particleNumber}" "${objectName}" "${sceneName}" "${rosbag}" "${repeat}" "${runAlgFlag}" "${ang_and_pos}" "${runVersion}" &
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