#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a objectNames=("cracker" "Ketchup")
# declare -a objectNames=("cracker" "Ketchup")
# declare -a objectNames=("cracker" "Ketchup" "Milk")
declare -a objectNames=("Parmesan")
# declare -a objectNames=("cracker" "Ketchup")
# declare -a objectNames=("Mustard")
# declare -a objectNames=("soup" "soup2")
# declare -a objectNames=("SaladDressing" "Mustard")
# declare -a objectNames=("cracker" "SaladDressing" "Parmesan")
# declare -a objectNames=("cracker" "Mayo" "Milk")
# declare -a objectNames=("cracker" "Ketchup" "Mayo" "Milk" "SaladDressing" "soup" "Parmesan" "Mustard")
# declare -a objectNames=("Parmesan" "soup")
# declare -a objectNames=("soup" "Parmesan" "Milk")
# declare -a objectNames=("Mayo")
# declare -a objectNames=("cracker" "Parmesan")
# declare -a objectNames=("Mustard" "SaladDressing")
# declare -a objectNames=("Mayo" "Milk")
# declare -a objectNames=("Ketchup")
# declare -a objectNames=("cracker" "gelatin" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a sceneNames=("scene6")

declare -a particleNumbers=(70)
# declare -a objectNames=("cracker")
# declare -a sceneNames=("scene3")
# declare -a runAlgFlags=("PBPF" "obse" "FOUD")
# declare -a runAlgFlags=("FOUD")
# declare -a runAlgFlags=("PBPF")
# declare -a runAlgFlags=("DiffDOPE")
# declare -a runAlgFlags=("DiffDOPE" "DiffDOPET")
# declare -a runAlgFlags=("FOUD" "DiffDOPE" "DiffDOPET")
# declare -a runAlgFlags=("PBPF" "obse")
declare -a runAlgFlags=("FOUD" "DiffDOPE")
# declare -a runobseFlags=("obse" "FOUD")
# declare -a Ang_and_Pos=("ang" "pos")
declare -a Ang_and_Pos=("ADD" "ADDS")
declare -a update_style_flag=("time") # "time" "pose"
# declare -a runVersions=("depth_img" "multiray")
# declare -a runVersions=("PBPF_RGBD" "PBPF_RGB" "PBPF_D")
declare -a runVersions=("PBPF_RGBD")
# declare -a massMarkers=("mA" "mB" "mC")
declare -a massMarkers=("mBNN")
# declare -a massMarkers=("mANN" "mA" "mBNN" "mB" "mCNN" "mC" "mDNN" "mD" "mENN" "mE" "mFNN" "mF" "mGNN" "mG")
# declare -a frictionMarkers=("fA" "fB" "fC" "fD")
declare -a frictionMarkers=("fB")
# declare -a frictionMarkers=("fA")
declare -a MF_flag=("mass") # mass/friction

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
						# for runobseFlag in "${runobseFlags[@]}"
						# do
						# 	python3 compute_err_after_align.py "${particleNumber}" "${objectName}" "${sceneName}" "${rosbag}" "${repeat}" "${runobseFlag}" "${ang_and_pos}" "${runVersion}" &
						# 	DATA_PRO_PID=$!
						# 	sleep 2
						# 	# for repeat in {1..10}
						for ((repeat=0;repeat<=0;repeat++));
						do
							for massMarker in "${massMarkers[@]}"
							do
								for frictionMarker in "${frictionMarkers[@]}"
								do

									for runVersion in "${runVersions[@]}"
									do
										for mf_flag in "${MF_flag[@]}"
										do
											python3 compute_err_after_align.py "${particleNumber}" "${objectName}" "${sceneName}" "${rosbag}" "${repeat}" "${runAlgFlag}" "${ang_and_pos}" "${runVersion}" "${massMarker}" "${frictionMarker}" "${mf_flag}" &
											DATA_PRO_PID=$!

											sleep 1
										done
									done

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
