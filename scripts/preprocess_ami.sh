#!/bin/bash

data_path=/data/xianghu/projects/KLASS/TSVAD_MC/data/ami
speaker_encoder_path=/data/xianghu/projects/KLASS/TSVAD_MC/pretrained_models/ecapa-tdnn.model

echo " Process AMI dataset, get json files"
python scripts/prepare_data.py \
    --data_path ${data_path} \
    --orig_audio_path ${data_path}/data_array01 \
    --rttm_path ${data_path}/rttm_array01 \
    --target_audio ${data_path}/target_audio_mc \
    --target_embedding ${data_path}/SpeakerEmbedding \
    --source ${speaker_encoder_path} \
