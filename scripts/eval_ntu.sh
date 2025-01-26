#!/bin/bash

gpu=0
exp_name=rerun_tsvad_mc_ali_ami_chime5
rs_len=4000
segment_shift=1
gen_subset=eval_bound

code_path=/data/xianghu/projects/KLASS/TSVAD_MC  # need change
data_path=/data/xianghu/projects/KLASS/TSVAD_MC/data/2023_ntu-recordings_16kv2  # need change

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=${code_path}/exp
ts_vad_path=${code_path}/ts_vad
speech_encoder_path=${code_path}/pretrained_models/ecapa-tdnn.model # Speaker encoder path
spk_path=${data_path}/SpeakerEmbedding

results_path=${exp_dir}/${exp_name}/inf # need change

python3 ${ts_vad_path}/generate.py ${data_path} \
  --user-dir ${ts_vad_path} \
  --results-path ${results_path} \
  --path ${exp_dir}/${exp_name}/checkpoints/checkpoint11.pt \
  --task ts_vad_mc \
  --dataset_name ntu \
  --spk-path ${spk_path} \
  --rs-len ${rs_len} \
  --segment-shift ${segment_shift} \
  --gen-subset ${gen_subset} \
  --batch-size 32 \
  --shuffle-spk-embed-level 3 \
  --inference
