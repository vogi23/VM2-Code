#!/bin/bash

# copy the jupyter configuration into home-directory
mkdir -p ~/.jupyter/
cp -r ${CONFIG_PATH}/jupyter_notebook_config.py ~/.jupyter/

mkdir -p ${PROJECT_PATH}

cd ${PROJECT_PATH}

jupyter lab