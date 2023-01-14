#!/bin/bash
# Mostly taken from stablebaselines3 at: https://github.com/DLR-RM/stable-baselines3/blob/master/scripts/build_docker.sh

CPU_PARENT=ubuntu:20.04
GPU_PARENT=nvidia/cuda:11.4.0-base-ubuntu20.04

TAG=szahlner/lcer
VERSION=$(cat ./lcer/version.txt)

if [[ ${USE_GPU} == "True" ]]; then
    PARENT=${GPU_PARENT}
    PYTORCH_DEPS="cudatoolkit=11.3"
else
    PARENT=${CPU_PARENT}
    PYTORCH_DEPS="cpuonly"
    TAG="${TAG}-cpu"
fi

echo "docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} . -f ./Docker/Dockerfile"
docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} . -f ./Docker/Dockerfile
docker tag ${TAG}:${VERSION} ${TAG}:latest

if [[ ${RELEASE} == "True" ]]; then
    docker push &{TAG}:${VERSION}
    docker push &{TAG}:latest
fi