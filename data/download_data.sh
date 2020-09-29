#!/bin/bash

# Install 7z
echo "Checking for 7z (might require installation)..."

if sudo apt-get install p7zip-full -y; then
    echo "7z found/installed"
else
    echo "Failed: Please install 7z (https://www.7-zip.org/7z.html)"
    exit 1
fi

# Download, Unzip, and Remove zip
if [ "$1" = "json" ]
then

    echo "Downloading JSONs ..."
    wget https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z
    7z x json_2.1.0.7z -y && rm json_2.1.0.7z
    echo "saved folder: json_2.1.0"

elif [ "$1" = "json_feat" ]
then

    echo "Downloading JSONs and Resnet18 feats ..."
    wget https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_feat_2.1.0.7z
    7z x json_feat_2.1.0.7z -y && rm json_feat_2.1.0.7z
    echo "saved folder: json_feat_2.1.0"

elif [ "$1" = "full" ]
then

    echo "Downloading JSONs, Resnet18 feats, Images, Videos, and PDDL states ..."
    wget https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/full_2.1.0.7z
    7z x full_2.1.0.7z -y && rm full_2.1.0.7z
    echo "saved folder: full_2.1.0"

else
    echo "Failed: Usage download_data.sh json | json_feat | full"
    exit 1
fi
