#!/bin/bash

# download files
mkdir data/
pushd data/
wget "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip"
wget "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip"

# unzip images
unzip -q '*.zip'

# move zip files to raw dir
mkdir raw/
mkdir processed/
mv *.zip raw/

# rename folders
mv images_background/ background/
mv images_evaluation/ evaluation/

# move 10 first evaluation subdirs to background dir
pushd evaluation/
folders=(*/)
popd
for ((i=0; i<10; i++))
do
  mv "evaluation/${folders[i]}" background/
done

mv background processed/
mv evaluation processed/

echo "done"
