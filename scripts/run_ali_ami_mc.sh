#!/bin/bash

gpu=0
config_name=base
exp_name=rerun_tsvad_mc_ali_ami
other_args=
code_path=/data/xianghu/projects/KLASS/TSVAD_MC  # need change
data_path=/data/xianghu/projects/KLASS/TSVAD_MC/data/alimeeting_ami  # need change

. ./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

exp_dir=${code_path}/exp
ts_vad_path=${code_path}/ts_vad
speech_encoder_path=${code_path}/pretrained_models/ecapa-tdnn.model # Speaker encoder path
spk_path=${data_path}/SpeakerEmbedding

mkdir -p ${exp_dir}/${exp_name}

HYDRA_FULL_ERROR=1 fairseq-hydra-train \
  --config-dir ${ts_vad_path}/config \
  --config-name ${config_name} \
  hydra.run.dir=${exp_dir}/${exp_name} \
  hydra.job.name=${exp_name} \
  hydra.sweep.dir=${exp_dir}/${exp_name} \
  task.data=${data_path} \
  common.user_dir=${ts_vad_path} \
  +task.spk_path=${spk_path} \
  +model.speech_encoder_path=${speech_encoder_path} \
  ${other_args}