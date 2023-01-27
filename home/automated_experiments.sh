#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a objectNames=("soup")
declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a particleNumbers=(35)

for particleNumber in "${particleNumbers[@]}"
do
	for objectName in "${objectNames[@]}"
	do
		
		for sceneName in "${sceneNames[@]}"
		do
			if [[ "$objectName" == "soup" ]]; then
				if [[ "$sceneName" == "scene4" ]]; then
					continue
				fi
			fi
			
			python3 update_yaml_file_automated.py "${objectName}" "${particleNumber}" "${sceneName}"
			
			for rosbag in {1..10}
			do
				for repeat in {1..10}
				do
					rosbag play "rosbag/new_rosbag/${objectName}_${sceneName}/${objectName}_${sceneName}_70_${rosbag}.bag" --clock &
					ROSBAGPID=$!
					
					rosrun PBPF Physics_Based_Particle_Filtering.py &
					PBPF_PID=$!
					
					fileName="${particleNumber}_${objectName}_${sceneName}_rosbag${rosbag}_repeat${repeat}_"
					rosrun PBPF RecordError.py "${fileName}" &
					REPID=$!
					
					sleep 40
					
					kill -SIGINT $REPID
					kill $ROSBAGPID
					kill -SIGINT $PBPF_PID
				done
			done

		done
	done
done
