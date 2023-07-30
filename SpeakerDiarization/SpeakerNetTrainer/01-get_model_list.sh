#!/bin/bash

ID=$1
DATASET=vox
SESSION_NUM=$2

nsml model ls $ID/${DATASET}/${SESSION_NUM}
