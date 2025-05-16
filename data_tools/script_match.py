
from sim_vsm import *
from sim_tokenvector import *

import json, os
import re


def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            res = json.load(f)
    except FileNotFoundError:
        res = []
    return res

def write_json(res, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        data = json.dumps(res, ensure_ascii=False, indent=2)
        f.write(data)

def stats_similarity(asr_path, date):
    simvsm = SimVsm()
    simtoken = SimTokenVec()

    dataset_script = load_json('xxx/SED_scrpits_cn.json')
    dialogues = [dataset_script[dk] for dk in dataset_script.keys() if date in dk]
    asr_json = load_json(asr_path)

    i = 0
    num_utterances = -1
    dialogue_id = ""
    for asr_ut in asr_json:
        text1 = asr_ut["asr"]
        print(text1)
        if i >= num_utterances:
            i = 0
            num_utterances = -1
            res = []
            for dialogue in dialogues:
                num_utterances = dialogue["num_utterances"]
                for utterance in dialogue["utterances"]:
                    text2 = utterance["text"]
                    idlist = utterance["utterance_id"].split('_')
                    dist = {
                        "dialogue_id": f'{idlist[0]}_{idlist[1]}',
                        "num_utterances": num_utterances,
                        "utterance_id": int(idlist[-1]),
                        "speaker_id": utterance["speaker_id"],
                        "simvsm": simvsm.distance(text1, text2),
                        "simtoken": simtoken.distance(text1, text2),
                        "text": text2
                    }
                    res.append(dist)
            res.sort(key=lambda x: x['simvsm'], reverse=True)
            # not match
            if res[0]["simvsm"] < 0.2:
                continue
            # match
            if res[0]["simvsm"] - res[1]["simvsm"] < 0.05 and res[0]["simtoken"] < res[1]["simtoken"]:
                top = res[1]
            else:
                top = res[0]
            num_utterances = top["num_utterances"]
            i = top["utterance_id"]
            dialogue_id = top["dialogue_id"]
            print(top["text"])
        else:
            print(dataset_script[dialogue_id]["utterances"][i]["text"])
        i += 2

if __name__ == "__main__":
    asr_path = "xxx"
    stats_similarity(asr_path, "250120")