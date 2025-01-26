import sys
import json

data_dir = "/data/xianghu/projects/KLASS/TSVAD_MC/data"
# json_1 = f"{data_dir}/alimeeting/Train_Ali/Train_Ali_far/Train.json"
# json_2 = f"{data_dir}/ami/Train.json"

json_1 = f"{data_dir}/alimeeting_ami/Train/Train.json"
json_2 = f"{data_dir}/chime5/Train.json"


f = open(f"{data_dir}/alimeeting_ami_chime5/Train/Train.json", "w")

lines_1 = open(json_1).read().splitlines()
lines_2 = open(json_2).read().splitlines()

for line in lines_1 + lines_2:
    data_json = json.loads(line)

    json.dump(data_json, f)
    f.write('\n')

    



