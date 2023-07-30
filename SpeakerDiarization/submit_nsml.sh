#!/bin/bash

# Usage:
#   bash submit_nsml.sh [ID] [SESSION_NUM] [DATASET]
#
# Example:
#   bash submit_nsml.sh KR20097 1 airush_sd_eval_1

ID=$1
SESSION_NUM=$2
DATASET=$3
CHECKPOINT=base

function build_session(){
    echo ${ID}/${DATASET}/${SESSION_NUM}
}

function submit_session(){
    local session="$(build_session ${ID} ${DATASET} ${SESSION_NUM})"

    nsml submit -v ${session} ${CHECKPOINT}
}

function submit_test_session(){
    local session="$(build_session ${ID} ${DATASET} ${SESSION_NUM})"

    nsml submit -v -t ${session} ${CHECKPOINT}
}

# If you want to submit with -t option, call submit_test_session instead.
submit_session
