#/bin/bash
# 
# change the following variables to set up code, input and output dirs
HOST_WORK_DIR="/home/local/zc13/Documents"
HOST_DATA_DIR="/home/mnt/data"

# get host user id to populate to the container
HOST_USER_ID="$(id -u)"
HOST_GROUP_ID="$(id -g)"
HOST_USER_NAME=${USER}
CONTAINER_WORKDIR="/workspace/Documents"   #/workspace/CTProjector" 
CONTAINER_DATADIR="/mnt/data" #should be /mnt/data when using NAS drive   

sudo docker run -it --runtime=nvidia --name=docker_ex --network="bridge" \
-v ${HOST_WORK_DIR}:${CONTAINER_WORKDIR} \
-v ${HOST_DATA_DIR}:${CONTAINER_DATADIR} \
-p 8100:8100 \
-e CONTAINER_UID=${HOST_USER_ID} \
-e CONTAINER_GID=${HOST_GROUP_ID} \
-e CONTAINER_UNAME=${HOST_USER_NAME} \
-e CONTAINER_WORKDIR=${CONTAINER_WORKDIR} \
zc_cuda11_tensorflow:1.0.0