#!/bin/bash

echo "Pulling repo"
git pull

echo "Starting Docker build process..."
docker build --no-cache -t lome/depthanythingsagemaker .
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 181172696166.dkr.ecr.us-east-2.amazonaws.com
docker tag lome/depthanythingsagemaker:latest 181172696166.dkr.ecr.us-east-2.amazonaws.com/lome/depthanythingsagemaker:latest
docker push 181172696166.dkr.ecr.us-east-2.amazonaws.com/lome/depthanythingsagemaker:latest

echo "Process completed successfully!"
echo
echo "Press any key to exit..."
read -n 1 -s -r