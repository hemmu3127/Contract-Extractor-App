#!/bin/bash

echo "Starting the Contract Extractor app"

# build a image with the same name and use the .sh file
# .sh file is applicable for linux or wsl only
docker run -it -d -p 8000:8000 contract_app:v1.0
