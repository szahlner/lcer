#!/bin/bash

CPU_PARENT=ubuntu:20.04
GPU_PARENT=nvidia/cuda:11.4.0-base-ubuntu20.04

TAG=lcer
VERSION=$(cat ./lcer/version.txt)

if [[ ${USE_GPU} == "True" ]]; then
  PARENT=${GPU_PARENT}
  PYTORCH_DEPS="--extra-index-url https://download.pytorch.org/whl/cu116"
else
  PARENT=${CPU_PARENT}
  PYTORCH_DEPS="cpuonly"
  TAG="${TAG}-cpu"
fi

echo "docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} ."
docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} .
docker tag ${TAG}:${VERSION} ${TAG}:latest