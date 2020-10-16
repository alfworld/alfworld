#!/bin/bash

# Download, Unzip, and Remove zip

cd $ALFRED_ROOT/data

# JSON files
rm 3cIEx9R*
wget https://bit.ly/3cIEx9R
mv 3cIEx9R 3cIEx9R.zip
unzip 3cIEx9R.zip
rm 3cIEx9R.zip

# PDDL files
rm 3mZsrhf*
wget https://bit.ly/3mZsrhf
mv 3mZsrhf 3mZsrhf.zip
unzip 3mZsrhf.zip
rm 3mZsrhf.zip

# TW-Game files
rm 2S8lexl*
wget https://bit.ly/2S8lexl
mv 2S8lexl 2S8lexl.zip
unzip 2S8lexl.zip
rm 2S8lexl.zip

# Pre-trained MaskRCNN model
wget https://aka.ms/alfworld/mrcnn.pth
mkdir -p $ALFRED_ROOT/agents/detector/models/
mv mrcnn.pth $ALFRED_ROOT/agents/detector/models/