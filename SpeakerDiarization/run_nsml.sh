#!/bin/bash

# Usage:
#   bash run_nsml.sh [CONFIG] [DATASET]
#
# Example:
#   bash run_nsml.sh configs/speaker_diarization.conf airush_sd_eval_1

GPU_SIZE=1
CHECKPOINT=base
ENTRY_POINT=main.py
MESSAGE="SpeakerDiarization"
CONFIG=$1
DATASET=$2

function build_argument(){
    local config=$1
    local checkpoint=$2

    echo "--config ${config} --checkpoint_name ${checkpoint}"
}

function run_session(){
    local config=$1
    local dataset=$2
    local argument="$(build_argument ${config} ${CHECKPOINT})"

    nsml run -g ${GPU_SIZE} \
             --gpu-driver-version 418.39 \
             -v \
             -e ${ENTRY_POINT} \
             -d ${DATASET} \
             -m ${MESSAGE} \
             -a "${argument}"
}

run_session ${CONFIG} ${DATASET}
