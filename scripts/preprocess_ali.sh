data_path=/data/xianghu/projects/KLASS/Multi-Channel-Speaker-Diarization/data/alimeeting

echo " Process dataset: Train/Eval dataset, get json files"
python scripts/prepare_data_ali.py \
    --data_path ${data_path} \
    --type Eval \

python scripts/prepare_data_ali.py \
    --data_path ${data_path} \
    --type Train \


