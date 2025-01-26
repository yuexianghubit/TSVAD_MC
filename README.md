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

### 2. Download AMI dataset
- Download AMI dataset from 

```
unzip ami.zip

bash scripts/preprocess_ami.sh
```


### 3. Merge Alimeeting and AMI dataset
```
cd data
mkdir alimeeting_ami
ln -s ../alimeeting/Eval_Ali . # we use the eval set of alimeeting for evaluation
ln -s ../../alimeeting/Train_Ali/Train_Ali_far/target_audio_mc/ . # link alimeeting data
ln -s ../../ami/target_audio_mc/* . # link ami data

python merge_json.py # merge alimeeting and ami data json
```


### 4. Download NTU recorded dataset
- Download NTU recorded dataset from https://1drv.ms/u/c/6c1f3d2be9b0f2f9/EflqL_U40xBBkkZJ9rxbcG0BMnbG0D899zTRkbV-kEhFKw?e=ua6OcD

```
tar -xvf 2023_ntu-recordings_16kv2.tar
cd 2023_ntu-recordings_16kv2
mkdir audio_bound
cp audio/*/*/*_chn00_bound.wav audio_bound # bound files are multi-channel
bash script/preprocess_ntu.sh
```


## Multi-Channel TS-VAD
### Train
```
bash scripts/run_ali_mc.sh # This will use only alimeeting dataset for training.
bash scripts/run_ali_ami_mc.sh # This will use alimeeting and ami datasets for training.
```
### Evaluate
```
bash scripts/eval_ali.sh # This will evaluate on alimeeting dataset.
bash scripts/eval_ntu.sh # This will evaluate on NTU recorded dataset.
```

### Results
| Model | Method | Training Data | Alimeeting Test | NTU Test |
|-------|--------|---------------|-----------------|----------|
| **#1** | Multi-Channel TS-VAD <br> (Speech Encoder: Ecapa TDNN) | Alimeeting | 3.86% | - |
| **#2** | Multi-Channel TS-VAD <br> (Speech Encoder: Ecapa TDNN) | Alimeeting + AMI | 3.66% | 39.21% |


### Pre-trained models
- The pre-trained model trained only on Alimeeting dataset can be obtained from https://1drv.ms/u/c/6c1f3d2be9b0f2f9/EQbrUxrufw9NoGXKeTWp0cABv4d-jinIXMDgy_mTMEOU4Q?e=HclC6e

- The pre-trained model trained only on Alimeeting and AMI datasets can be obtained from https://1drv.ms/u/c/6c1f3d2be9b0f2f9/EWpA9_PNYm1IkEz6D8ikV6cBFKJ2icKz38UVmwJwFdxIdQ?e=jFlpuu

