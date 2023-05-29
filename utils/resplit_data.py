import json
import random

if __name__ == "__main__":
    ann_path = 'C:/wrd/ss304/ss304/train/train.json'
    with open(ann_path) as f:
        data = json.load(f)
    tmp = []
    for key, value in data.items():
        tmp.append([key, value])
    index = list(range(len(tmp)))
    random.shuffle(index)
    random.shuffle(index)
    index = index[:len(tmp) // 20]
    ret = {}
    for i in index:
        ret.update({tmp[i][0]: tmp[i][1]})
    json_data = json.dumps(ret, indent=4)
    with open('fe-few-shot-train.json', 'w') as f:
        f.write(json_data)
