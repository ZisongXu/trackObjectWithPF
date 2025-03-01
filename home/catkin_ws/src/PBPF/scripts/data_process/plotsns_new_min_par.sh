#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a objectNames=("cracker" "Ketchup")
# declare -a objectNames=("cracker" "gelatin" "soup")
# declare -a objectNames=("soup" "Parmesan" "Milk")
declare -a objectNames=("soup" "soup2")
# declare -a objectNames=("SaladDressing" "Mustard" "Mayo")
# declare -a objectNames=("Mustard" "SaladDressing")
# declare -a objectNames=("cracker" "Ketchup" "Milk")
# declare -a objectNames=("cracker" "soup")
# declare -a objectNames=("cracker")
# declare -a objectNames=("cracker" "Ketchup" "Mayo" "Milk" "Mustard" "Parmesan" "SaladDressing" "soup")
# declare -a objectNames=("cracker" "Ketchup")
# declare -a objectNames=("SaladDressing" "soup")
# declare -a objectNames=("SaladDressing" "Mustard")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
# declare -a objectNames=("Mayo" "Milk")
declare -a sceneNames=("scene2")
# declare -a objectNames=("cracker" "soup" "Parmesan")
# declare -a objectNames=("cracker" "Mayo" "Milk")
# declare -a objectNames=("Ketchup" "Mayo" "Milk" "SaladDressing" "soup" "Parmesan" "Mustard")

declare -a particleNumbers=(50)
# declare -a objectNames=("cracker")
# declare -a sceneNames=("scene3")

# declare -a Ang_and_Pos=("pos" "ang")
declare -a Ang_and_Pos=("ADD" "ADDS")
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

						# python3 plotsns_new_min_par.py "${particleNumber}" "${objectName}" "${sceneName}" "${rosbag}" "${update_style}" "${ang_and_pos}" &
						python3 plotsns_new_avg_par.py "${particleNumber}" "${objectName}" "${sceneName}" "${rosbag}" "${update_style}" "${ang_and_pos}" &
					# PLOT_PID=$!
					# sleep 100
					done
				done
			done
		done
	done
done
