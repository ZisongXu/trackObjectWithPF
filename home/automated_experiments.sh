#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a objectNames=("cracker")
declare -a sceneNames=("scene2")
declare -a particleNumbers=(15 20 25 30)
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
				# for ((rosbag=6;rosbag<=6;rosbag++)); 
				do
					duration=$(python3 get_info_from_rosbag.py "${objectName}" "${particleNumber}" "${sceneName}" "${rosbag}")

					for repeat in {1..10}
					# for ((repeat=5;repeat<=5;repeat++));
					do
						echo "I will sleep for $duration seconds"
						rosbag play "rosbag/new_rosbag/${objectName}_${sceneName}/${objectName}_${sceneName}_70_${rosbag}.bag" --clock  > /dev/null 2>&1 & 
						ROSBAGPID=$!

						rosrun PBPF Physics_Based_Particle_Filtering.py &
						PBPF_PID=$!

						sleep 10
						
						fileName="${particleNumber}_${objectName}_${sceneName}_rosbag${rosbag}_repeat${repeat}_"
						rosrun PBPF RecordError.py "${fileName}" &
						REPID=$!
						
						sleep $duration
						kill -SIGINT $REPID
						sleep 5
						pkill -9 RecordE*
						kill -SIGINT $PBPF_PID
						pkill -9 Physics_Based_*
						sleep 5
					done
				done

			done
		done
	done
done
