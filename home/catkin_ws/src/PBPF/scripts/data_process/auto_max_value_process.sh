#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a objectNames=("cracker")
declare -a sceneNames=("scene1" "scene2" "scene3")
declare -a particleNumbers=(1 5 10 15 20 70)
# declare -a runAlgFlags=("PBPF" "CVPF")
declare -a repeatTimes=(10)
declare -a Ang_and_Pos=("pos")
declare -a filenames=("max_value")

for particleNumber in "${particleNumbers[@]}"
do
	for objectName in "${objectNames[@]}"
	do
		for sceneName in "${sceneNames[@]}"
		do
			for Ang_a_Pos in "${Ang_and_Pos[@]}"
			do
				python3 take_max_error_value.py "${particleNumber}" "${objectName}" "${sceneName}" "${Ang_a_Pos}" &
				DATA_PRO_PID=$!

				sleep 4
			done
		done
	done
done

for filename in "${filenames[@]}"
do

	python3 plotsns_base_on_parNum.py "${filename}" &
	DATA_PRO_PID=$!

done
