#!/bin/bash


data_path=/data/xianghu/projects/KLASS/TSVAD_MC/data/2023_ntu-recordings_16kv2
speaker_encoder_path=/data/xianghu/projects/KLASS/TSVAD_MC/pretrained_models/ecapa-tdnn.model

echo " Process NTU dataset, get json files"
python scripts/prepare_data.py \
    --data_path ${data_path} \
    --orig_audio_path ${data_path}/audio_bound \
    --rttm_path ${data_path}/rttm_bound \
    --target_audio ${data_path}/target_audio_bound_mc \
    --target_embedding ${data_path}/SpeakerEmbedding \
    --type eval_bound \
    --source ${speaker_encoder_path} \
