#!/bin/bash

###############################################################################
# retrain.sh
###############################################################################
#
# Description: Runs the tensorflow docker container's retrain.py
# to build another layer with catagories for the pokemon in the 
# tf_files/pokemon folder. 
#
###############################################################################
#
# Created By: Deepak Selvan
# Created Date: 02.19.17
# Last Updated By: 
# Last Modified Date: 
# Version v1.0.0
# 
###############################################################################

# Make sure you are in the tf_files dir to mount the folder into the container
cd "$(dirname ${BASH_SOURCE[0]})"

# Capture UID to inject into docker container 
if [[ -z "${UID}" ]]; then UID=$(id -u "${USER}"); fi

# Docker run to generate the retrained graph using the files in 
# tf_files/pokemon.
sudo docker run -t \
    -v "$(pwd)":/tf_files \
    -w /tensorflow \
     tensorflow/tensorflow:latest-devel \
     bash -c "useradd -r -u ${UID} ${USER} && \
        su ${USER} -c 'python tensorflow/examples/image_retraining/retrain.py \
            --bottleneck_dir=/tf_files/bottlenecks --how_many_training_steps 500 \
            --model_dir=/tf_files/inception \
            --output_graph=/tf_files/retrained_graph.pb \
            --output_labels=/tf_files/retrained_labels.txt \
            --image_dir /tf_files/pokemon/'"