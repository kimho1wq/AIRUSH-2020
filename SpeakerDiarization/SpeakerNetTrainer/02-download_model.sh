#!/bin/bash

ID=$1
DATASET=vox
SESSION_NUM=$2
CHECKPOINT=$3

nsml model pull $ID/${DATASET}/${SESSION_NUM} ${CHECKPOINT} ./
mv ${ID}_${DATASET}_${SESSION_NUM}/${CHECKPOINT}/model/pretrained.pt ../SpeakerDiarization/third_party/SpeakerNet/models/weights/16k/pretrained.model
