#!/bin/bash
# --contain without binding /dev was causing some strange CPU usage whenever MuJoCo
# was running, two solutions found: not using --contain at all, or using --contain and
# binding host's /dev to container.

rm -rf workdir/*

singularity run \
    --contain \
    $(if lspci | grep -qi nvidia; then echo ' --nv'; else echo ''; fi) \
    --bind=/etc/hosts \
    --bind=/etc/localtime \
    --workdir=workdir \
    --bind=/proc \
    --bind=/sys \
    --bind=/dev \
    --bind=scripts:/scripts \
    --home=home:${HOME} \
    trackObjectWithPF
