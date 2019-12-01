#!bin//bash

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

docker stop api-server
docker rm api-server

docker build -t api-img .

docker run --name api-server \
    -p 80:3000\
    -d \
    api-img