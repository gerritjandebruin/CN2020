#!/bin/bash

rsync -aPhi --delete /home/bruingjde/complexnetworks2020-experiment/ nieuw-engeland.nl:/volume1/backup1/liacs/complexnetworks2020-experiment/
git add --all
git commit -m "WIP"
git push