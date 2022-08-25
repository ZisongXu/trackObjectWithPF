#!/bin/bash
# --contain without binding /dev was causing some strange CPU usage whenever MuJoCo
# was running, two solutions found: not using --contain at all, or using --contain and
# binding host's /dev to container.

cd rosbag_info
