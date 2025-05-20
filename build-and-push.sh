#!/bin/bash

echo "Pulling repo"
git pull

echo "Starting Docker build process..."
docker build -t lome/depthanythingsagemaker-newarch .
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 181172696166.dkr.ecr.us-east-2.amazonaws.com
docker tag lome/depthanythingsagemaker-newarch:latest 181172696166.dkr.ecr.us-east-2.amazonaws.com/lome/depthanythingsagemaker-newarch:latest
docker push 181172696166.dkr.ecr.us-east-2.amazonaws.com/lome/depthanythingsagemaker-newarch:latest

echo "Process completed successfully!"
echo
echo "Press any key to exit..."
read -n 1 -s -r