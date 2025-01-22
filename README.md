# TSVAD_MC

## Install
```
conda env create -f environment.yml
```


## Dataset
### 1. Download Alimeeting dataset (https://openslr.org/119/)
```
mkdir alimeeting
cd alimeeting
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_far.tar.gz
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz
tar -xzvf Train_Ali_far.tar.gz
tar -xzvf Eval_Ali.tar.gz
```
- Download speaker model (ecapa-tdnn) from https://1drv.ms/u/c/6c1f3d2be9b0f2f9/EWJeXgE8KEtCp9nVUlCNnoYBDef3nMSk2XsHBoBEtILXpg?e=YtJEvl
- Download speaker embedding from https://drive.google.com/file/d/1tNRnF9ouPbPX9jxAh1HkuNBOY9Yx6Pj9/view?usp=sharing

- Then make the dataset look like:
 ```
alimeeting
├── Train_Ali
│   ├── Train_Ali_far 
│     ├── audio_dir
├── Eval_Ali
│   ├── Eval_Ali_far 
│     ├── audio_dir
├── spk_embed
│   ├── SpeakerEmbedding 
│     ├── ...
```

- run scripts/preprocess_ali.sh and then the dataset should look like
```
alimeeting 
├── Train_Ali
│   ├── Train_Ali_far 
│     ├── audio_dir
│     ├── target_audio
│     ├── textgrid_dir
│     ├── Train.json
├── Eval_Ali
│   ├── Eval_Ali_far 
│     ├── audio_dir
│     ├── target_audio
│     ├── textgrid_dir
│     ├── Eval.json
├── spk_embed
│   ├── SpeakerEmbedding 
│     ├── ...
```

## Multi-Channel TS-VAD
### Train
```
bash scripts/run_ali.sh # This will use only alimeeting dataset for training.
bash scripts/run_ali_ami.sh # This will use alimeeting and ami datasets for training.
```
### Evaluate
```
bash scripts/eval_ali.sh # This will evaluate on alimeeting dataset.
bash scripts/eval_ntu.sh # This will evaluate on NTU recorded dataset.
```


