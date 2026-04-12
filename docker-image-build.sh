#!/bin/bash
# build image for docker hub

versionFile="VERSION"
if [[ ! -f ${versionFile} ]]; then
    echo "Version file not found!"
    exit 1
fi

version=$(cat ${versionFile})
docker build -f Dockerfile \
  -t nullata/llamaman:latest \
  -t nullata/llamaman:${version} .
