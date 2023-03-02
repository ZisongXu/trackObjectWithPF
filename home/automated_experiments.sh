#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a objectNames=("cracker")
declare -a sceneNames=("scene1")
declare -a particleNumbers=(70)
declare -a runAlgFlags=("PBPF")
declare -a diffRadSigma=(0.32505 0.2167)
declare -a repeats=(8)

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
				
				# for rosbag in {1..10}
				# for rosbag in {1..2}
				for ((rosbag=2;rosbag<=2;rosbag++)); 
				do
					duration=$(python3 get_info_from_rosbag.py "${objectName}" "${particleNumber}" "${sceneName}" "${rosbag}")

					# for repeat in {1..10}
					# for repeat in "${repeats[@]}"
					for ((repeat=1;repeat<=1;repeat++));
					do
						echo "I will sleep for $duration seconds"
						rosbag play "rosbag/latest_rosbag/${objectName}_${sceneName}/${objectName}_${sceneName}_70_${rosbag}.bag" --clock  > /dev/null 2>&1 & 
						ROSBAGPID=$!

						rosrun PBPF Physics_Based_Particle_Filtering.py &
						PBPF_PID=$!

						sleep 14
						
						fileName="${particleNumber}_${objectName}_${sceneName}_rosbag${rosbag}_repeat${repeat}_"
						rosrun PBPF RecordError.py "${fileName}" &
						REPID=$!
						
						sleep $duration
						kill -SIGINT $REPID
						sleep 2
						pkill -9 RecordE*
						kill -SIGINT $PBPF_PID
						pkill -9 Physics_Based_*
						sleep 2
					done
				done

			done
		done
	done
done
