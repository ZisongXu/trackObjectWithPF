#!/bin/bash

# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
# declare -a objectNames=("cracker" "gelatin" "soup")
declare -a objectNames=("cracker")
declare -a sceneNames=("scene2")
declare -a particleNumbers=(70)
declare -a runAlgFlags=("PBPF")
declare -a diffRadSigma=(0.32505 0.2167)
declare -a repeats=(1)
# declare -a runVersions=("depth_img" "multiray")
declare -a runVersions=("PBPF_RGBD" "PBPF_RGB" "PBPF_D")



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

				
				# for rosbag in {1..10}
				# for rosbag in {1..2}
				for ((rosbag=1;rosbag<=1;rosbag++)); 
				do
					for runVersion in "${runVersions[@]}"
					do
						
						python3 update_yaml_file_automated.py "${objectName}" "${particleNumber}" "${sceneName}" "${runAlgFlag}" "${runVersion}" 

						duration=$(python3 get_info_from_rosbag.py "${objectName}" "${particleNumber}" "${sceneName}" "${rosbag}")

						# for repeat in {1..10}
						# for repeat in "${repeats[@]}"
						for ((repeat=0;repeat<=2;repeat++));
						do
							echo "I will sleep for $duration seconds"
							# rosbag play "rosbag/latest_rosbag/${objectName}_${sceneName}/${objectName}_${sceneName}_70_${rosbag}.bag" --clock  > /dev/null 2>&1 & 
							rosbag play "rosbag/depth_image_cracker_soup_barry${rosbag}.bag" --clock --rate 0.01  > /dev/null 2>&1 & 
							ROSBAGPID=$!

							rosrun PBPF Physics_Based_Particle_Filtering.py &
							PBPF_PID=$!

							# sleep 14
							
							# fileName="${particleNumber}_${objectName}_${sceneName}_rosbag${rosbag}_repeat${repeat}_"
							fileName="${particleNumber}_${sceneName}_rosbag${rosbag}_repeat${repeat}_"
							rosrun PBPF RecordError.py "${fileName}" &
							REPID=$!
							
							sleep $duration
							kill -SIGINT $REPID
							sleep 10
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
done