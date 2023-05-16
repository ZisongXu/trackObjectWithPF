#!/bin/bash

# declare -a objectNames=("cracker" "soup")
# declare -a sceneNames=("scene1" "scene2" "scene3" "scene4")
# declare -a objectNames=("cracker")
# declare -a sceneNames=("scene1")
# declare -a particleNumbers=(70)
# declare -a runAlgFlags=("PBPF")
# declare -a diffRadSigma=(0.32505 0.2167)
# declare -a repeats=(8)


rosrun PBPF Physics_Based_Particle_Filtering.py &

rosrun PBPF Visualisation_World_DOPE.py &

rosrun PBPF Visualisation_World_PBPF.py &
	