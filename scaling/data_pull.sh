#!/bin/bash

# Uses rsync to pull down any changes made to the target directory
# on graham; pauses for 1 hour between each rsync call since the 
# model training process takes a long time
while :
do
    sleep 1h

    rsync -arze \
    ssh graham:/home/ejaazm/projects/def-rgmelko/ejaazm/model_snapshots \
    /home/emerali/scaling_studies/tfim1d/model_snapshots
done