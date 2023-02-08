#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
declare -a objectNames=("cracker" "soup")
declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
# declare -a particleNumbers=(70)
# declare -a runAlgFlags=("PBPF" "CVPF")
declare -a repeatTimes=(10)
declare -a Ang_and_Pos=("ang" "pos")


for repeat in {1..10}
do

	python3 data_processing.py "${repeat}" &
	DATA_PRO_PID=$!

	sleep 250

done


for style in "${Ang_and_Pos[@]}"
do
	python3 inter_data.py "${style}" &
	INTER_DATA_PID=$!
	sleep 5

	python3 plotsns.py "${style}" &
	PLOT_PID=$!
	sleep 100
done 
