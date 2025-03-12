#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a objectNames=("cracker" "Ketchup")
# declare -a objectNames=("cracker" "gelatin" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
# declare -a objectNames=("soup" "Parmesan")
# declare -a objectNames=("SaladDressing" "Mustard")
# declare -a objectNames=("Mayo" "Milk")
# declare -a objectNames=("cracker" "Ketchup")
# declare -a objectNames=("SaladDressing")
# declare -a objectNames=("SaladDressing")
declare -a objectNames=("Parmesan")
# declare -a objectNames=("cracker" "Ketchup" "Mayo" "Milk" "Mustard" "Parmesan" "SaladDressing")
# declare -a sceneNames=("scene1" "scene2")
declare -a sceneNames=("scene6")
# declare -a objectNames=("cracker" "soup" "Parmesan")
# declare -a objectNames=("cracker" "Mayo" "Milk")
# declare -a objectNames=("Ketchup" "Mayo" "Milk" "SaladDressing" "soup" "Parmesan" "Mustard")


declare -a particleNumbers=(70)
# declare -a objectNames=("cracker")
# declare -a sceneNames=("scene3")
declare -a runAlgFlags=("PBPF")
# declare -a Ang_and_Pos=("ang" "pos")
declare -a Ang_and_Pos=("ADD")
declare -a update_style_flag=("time") # "time" "pose"
# declare -a runVersions=("depth_img" "multiray")
# declare -a runVersions=("PBPF_D" "PBPF_RGB" "PBPF_RGBD")
declare -a runVersions=("PBPF_RGBD")
# declare -a runVersions=("PBPF_Opti" "PBPF_RGB")
# declare -a massMarkers=("mA" "mB" "mC" "mD")
# declare -a massMarkers=("mANN" "mBNN" "mCNN" "mDNN" "mA" "mB" "mC" "mD" "mAN" "mBN" "mCN" "mDN")
# declare -a massMarkers=("mANN" "mA" "mBNN" "mB" "mCNN" "mC" "mDNN" "mD" "mENN" "mE" "mFNN" "mF" "mGNN" "mG")
# declare -a frictionMarkers=("fANN" "fBNN" "fCNN" "fDNN" "fENN" "fFNN" "fE" "fF")
# declare -a massMarkers=("mAN" "mBN" "mCN" "mDN")
# declare -a massMarkers=()
# declare -a massMarkers=("mANN" "mBNN" "mCNN" "mDNN" "mENN" "mFNN" "mGNN")
declare -a massMarkers=("mBNN")
# declare -a massMarkers=("mANN" "mBNN" "mCNN" "mDNN" "mENN")
# declare -a frictionMarkers=("fA" "fB" "fC" "fD" "fAN" "fBN" "fCN" "fDN")
# declare -a frictionMarkers=("fANN" "fBNN" "fCNN" "fDNN" "fENN" "fFNN" "fA" "fB" "fC" "fD" "fE" "fF")
# declare -a frictionMarkers=("fA" "fB" "fC" "fD" "fE" "fF" "fG" "fH" "fAN" "fBN" "fCN" "fDN" "fEN" "fFN" "fGN" "fHN" "fANN" "fBNN" "fCNN" "fDNN" "fENN" "fFNN" "fGNN" "fHNN")
declare -a frictionMarkers=("fB")
# declare -a massMarkers=("mB")
# declare -a frictionMarkers=("fANN" "fA" "fBNN" "fB" "fCNN" "fC" "fDNN" "fD" "fENN" "fE" "fFNN" "fF" "fGNN" "fG")

for objectName in "${objectNames[@]}"
do
	for ang_and_pos in "${Ang_and_Pos[@]}"
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
						for ((repeat=0;repeat<=4;repeat++));
						do
							for massMarker in "${massMarkers[@]}"
							do
								for frictionMarker in "${frictionMarkers[@]}"
								do
									for runVersion in "${runVersions[@]}"
									do
										for ((par_index=0;par_index<${particleNumber};par_index++)); 
										do
											python3 split_data.py "${particleNumber}" "${sceneName}" "${rosbag}" "${repeat}" "${runAlgFlag}" "${runVersion}" "${par_index}" "${objectName}" "${massMarker}" "${frictionMarker}" &
											DATA_PRO_PID=$!

											sleep 0.1
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
