#!/bin/bash
# cracker soup Ketchup Mayo Milk
# declare -a objectNames=("cracker" "soup")
# declare -a objectNames=("Parmesan" "Milk")
# declare -a objectNames=("Parmesan" "Mayo")
# declare -a objectNames=("Parmesan" "Mustard")
# declare -a objectNames=("Parmesan" "SaladDressing")
# declare -a objectNames=("cracker" "SaladDressing")
# declare -a objectNames=("Parmesan" "Milk")
# declare -a objectNames=("Parmesan" "Ketchup")
# declare -a objectNames=("Mustard" "Parmesan")
# declare -a objectNames=("SaladDressing" "Mustard" "Mayo")
# declare -a objectNames=("SaladDressing" "soup")
# declare -a objectNames=("Ketchup" "Parmesan")
# declare -a objectNames=("Parmesan" "Ketchup")
# declare -a objectNames=("SaladDressing" "Mustard")
# declare -a objectNames=("cracker" "soup")
# declare -a objectNames=("SaladDressing" "Mustard" "Mayo")
# declare -a objectNames=("soup" "Mayo")
# declare -a objectNames=("Mayo")
# declare -a objectNames=("Mustard")
# declare -a objectNames=("Milk")
# declare -a objectNames=("Milk" "Mustard")
# declare -a objectNames=("Mustard" "SaladDressing")
declare -a objectNames=("cracker" "Ketchup" "Mayo" "Milk" "Mustard" "Parmesan" "SaladDressing" "soup")
# declare -a objectNames=("Ketchup" "Mayo" "Milk" "SaladDressing" "soup" "Parmesan" "Mustard")

# declare -a objectNames=("cracker" "gelatin" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
# declare -a sceneNames=("scene1")
declare -a sceneNames=("scene2")

declare -a particleNumbers=(1)
# declare -a objectNames=("cracker")
# declare -a sceneNames=("scene3")
# declare -a runAlgFlags=("GT" obse)
declare -a runAlgFlags=("GT")
# declare -a runAlgFlags=("FOUD")
# declare -a runAlgFlags=("GT" "FOUD")
# declare -a runAlgFlags=("obse" "PBPF" "GT")
# declare -a runAlgFlags=("GT" "FOUD")
# declare -a Ang_and_Pos=("ang" "pos")
declare -a Ang_and_Pos=("ADD")
declare -a update_style_flag=("time") # "time" "pose"
# declare -a runVersions=("depth_img" "multiray")
# declare -a runVersions=("PBPF_RGBD" "PBPF_RGB" "PBPF_D")
declare -a runVersions=("PBPF_RGBD")
# declare -a runVersions=("PBPF_Opti" "PBPF_RGB")
# declare -a massMarkers=("mA" "mB" "mC" "mD" "mAN" "mBN" "mCN" "mDN")
# declare -a massMarkers=("mANN" "mBNN" "mCNN" "mDNN" "mA" "mB" "mC" "mD" "mAN" "mBN" "mCN" "mDN")
# declare -a massMarkers=("mANN" "mA" "mBNN" "mB" "mCNN" "mC" "mDNN" "mD" "mENN" "mE" "mFNN" "mF" "mGNN" "mG")
declare -a massMarkers=("mBNN")
# declare -a massMarkers=("mANN" "mBNN" "mCNN" "mDNN" "mENN" "mFNN" "mGNN")
# declare -a massMarkers=("mANN" "mBNN" "mCNN" "mDNN" "mENN")
declare -a frictionMarkers=("fB")
# declare -a massMarkers=("mA" "mB" "mC" "mD")
# declare -a massMarkers=("mA")
# declare -a frictionMarkers=("fA" "fB" "fC" "fD")
# declare -a massMarkers=("mA")
# declare -a frictionMarkers=("fA" "fB" "fC" "fD" "fAN" "fBN" "fCN" "fDN")
# declare -a frictionMarkers=("fANN" "fBNN" "fCNN" "fDNN" "fENN" "fFNN" "fA" "fB" "fC" "fD" "fE" "fF")
# declare -a massMarkers=("mB")
# declare -a frictionMarkers=("fANN" "fA" "fBNN" "fB" "fCNN" "fC" "fDNN" "fD" "fENN" "fE" "fFNN" "fF" "fGNN" "fG")
# declare -a frictionMarkers=("fA" "fB" "fC" "fD" "fAN" "fBN" "fCN" "fDN")
# declare -a frictionMarkers=("fA" "fB" "fC" "fD" "fE" "fF" "fG" "fH" "fAN" "fBN" "fCN" "fDN" "fEN" "fFN" "fGN" "fHN" "fANN" "fBNN" "fCNN" "fDNN" "fENN" "fFNN" "fGNN" "fHNN")

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
							for massMarker in "${massMarkers[@]}"
							do
								for frictionMarker in "${frictionMarkers[@]}"
								do
									for runVersion in "${runVersions[@]}"
									do
										python3 align_time_data_process.py "${particleNumber}" "${objectName}" "${sceneName}" "${rosbag}" "${repeat}" "${runAlgFlag}" "${ang_and_pos}" "${runVersion}" "${massMarker}" "${frictionMarker}" &
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
