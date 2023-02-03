#!/bin/bash
# Mostly taken from stablebaselines3 at: https://github.com/DLR-RM/stable-baselines3/blob/master/scripts/build_docker.sh

CPU_PARENT=ubuntu:20.04
GPU_PARENT=nvidia/cuda:11.3.0-devel-ubuntu20.04

TAG=szahlner/lcer
VERSION=$(cat ./lcer/version.txt)

if [[ ${USE_GPU} == "True" ]]; then
    PARENT=${GPU_PARENT}
    PYTORCH_INSTALL="pip install --no-input torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113"
    FAISS_INSTALL="conda install -c pytorch faiss-gpu cudatoolkit=11.3"
else
    PARENT=${CPU_PARENT}
    PYTORCH_INSTALL="pip install --no-input torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu"
    FAISS_INSTALL="conda install -c pytorch faiss-cpu"
    TAG="${TAG}-cpu"
fi

echo "docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_INSTALL=${PYTORCH_INSTALL} --build-arg FAISS_INSTALL=${FAISS_INSTALL} -t ${TAG}:${VERSION} . -f ./Docker/Dockerfile"
docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_INSTALL=${PYTORCH_INSTALL} --build-arg FAISS_INSTALL=${FAISS_INSTALL} -t ${TAG}:${VERSION} . -f ./Docker/Dockerfile
docker tag ${TAG}:${VERSION} ${TAG}:latest

if [[ ${RELEASE} == "True" ]]; then
    docker push &{TAG}:${VERSION}
    docker push &{TAG}:latest
fi