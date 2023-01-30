#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a objectNames=("soup")
declare -a sceneNames=("scene2")
declare -a particleNumbers=(52)
declare -a runAlgFlags=("PBPF")

for runAlgFlag in "${runAlgFlags[@]}"
do
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
				
				python3 update_yaml_file_automated.py "${objectName}" "${particleNumber}" "${sceneName}" "${runAlgFlag}"
				
				for rosbag in {1..10}
				# for ((rosbag=7;rosbag<=7;rosbag++)); 
				do
					for repeat in {1..10}
					# for ((repeat=2;repeat<=2;repeat++));
					do
						if [[ ! -f "${particleNumber}_${objectName}_${sceneName}_rosbag${rosbag}_repeat${repeat}_time_PBPF_err_pos.csv"]]
						then
							echo "This file ${particleNumber}_${objectName}_${sceneName}_rosbag${rosbag}_repeat${repeat}_time_PBPF_err_pos.csv does not exists"
						fi
						
						rosbag play "rosbag/new_rosbag/${objectName}_${sceneName}/${objectName}_${sceneName}_70_${rosbag}.bag" --clock  > /dev/null 2>&1 & 
						ROSBAGPID=$!
						
						rosrun PBPF Physics_Based_Particle_Filtering.py &
						PBPF_PID=$!
						
						fileName="${particleNumber}_${objectName}_${sceneName}_rosbag${rosbag}_repeat${repeat}_"
						rosrun PBPF RecordError.py "${fileName}" &
						REPID=$!
						
						sleep 50
						
						kill -SIGINT $REPID
						kill -SIGINT $PBPF_PID
						pkill -9 Physics_Based_*

						sleep 10

						pkill -9 RecordE*

						sleep 5
						
						
					done
				done

			done
		done
	done
done
