#!/bin/bash

echo "Checking if docker is installed"
if ! [ -x "$(command -v docker)" ]; then
    echo "Install and start docker"
    yum update -y
    yum install -y docker
    service docker start
    usermod -a -G docker ec2-user
else
    echo 'Docker is installed'
fi

# Stopping previos dockers containers
docker stop mongodb
docker rm mongodb

# Bringing mongo
docker pull mongo

docker run \
    --name mongodb \
    -d \
    -p 27017-27019:27017-27019  \
    -v ~/mongo/data:/data/db \
    mongo 