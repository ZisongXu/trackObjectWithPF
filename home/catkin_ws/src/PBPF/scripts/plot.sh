#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a objectNames=("cracker" "soup")
declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
# declare -a particleNumbers=(70)
# declare -a runAlgFlags=("PBPF" "CVPF")
declare -a repeatTimes=(10)
declare -a Ang_and_Pos=("ang" "pos")

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
for repeat in {1..10}
do

	python3 data_processing.py "${repeat}" &
	DATA_PRO_PID=$!

	sleep 250
	# rosbag play "rosbag/new_rosbag/${objectName}_${sceneName}/${objectName}_${sceneName}_70_${rosbag}.bag" --clock  > /dev/null 2>&1 & 
	# ROSBAGPID=$!
	
	# rosrun PBPF Physics_Based_Particle_Filtering.py &
	# PBPF_PID=$!
	
	# fileName="${particleNumber}_${objectName}_${sceneName}_rosbag${rosbag}_repeat${repeat}_"
	# rosrun PBPF RecordError.py "${fileName}" &
	# REPID=$!
	
	# sleep 50
	
	# kill -SIGINT $REPID
	# kill -SIGINT $PBPF_PID
	# pkill -9 Physics_Based_*

	# sleep 10

	# pkill -9 RecordE*

	# sleep 5
done
# 				done

# 			done
# 		done
# 	done
# done

for style in "${Ang_and_Pos[@]}"
do
	python3 inter_data.py "${style}" &
	INTER_DATA_PID=$!
	sleep 5

	python3 plotsns.py "${style}" &
	PLOT_PID=$!
	sleep 100
done 
