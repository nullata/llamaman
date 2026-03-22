#!/bin/bash
# build image for docker hub

versionFile="VERSION"
if [[ ! -f ${versionFile} ]]; then
    echo "Version file not found!"
    exit 1
fi



version=$(cat ${versionFile})
docker build -f Dockerfile.cuda \
  -t nullata/llamaman:cuda-latest \
  -t nullata/llamaman:cuda-${version} .

docker build -f Dockerfile.rocm \
  -t nullata/llamaman:rocm-latest \
  -t nullata/llamaman:rocm-${version} .
