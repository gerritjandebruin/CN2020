#!/bin/bash

git status
git add --all

read -p "Commit? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
  git commit -m "WIP"
  git push
fi


rsync -aPhi --delete /home/bruingjde/complexnetworks2020-experiment/ nieuw-engeland.nl:/volume1/backup1/liacs/complexnetworks2020-experiment/