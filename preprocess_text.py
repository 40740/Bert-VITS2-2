import json
import os
from random import shuffle

# download url: "https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz"
os.environ["OPEN_JTALK_DICT_DIR"] = "./download/open_jtalk_dic_utf_8-1.11"
from text.cleaner import clean_text
from collections import defaultdict
import tqdm

stage = [1, 2, 3]

transcription_paths = []
for name in os.listdir("./annotations"):
    if not (name.endswith(".txt") and name.startswith("anno_")):
        continue
    transcription_paths.append(os.path.join("./annotations", name))
train_path = "annotations/train.list"
val_path = "annotations/val.list"
config_dir = "configs"
val_per_spk = 4
max_val_total = 8

if 1 in stage:
    for transcription_path in transcription_paths:
        with open(transcription_path + ".cleaned", "w", encoding="utf-8") as f:
            for line in tqdm.tqdm(
                open(transcription_path, encoding="utf-8").readlines()
            ):
                try:
                    utt, spk, language, text = line.strip().split("|")
                    # language = "ZH"
                    norm_text, phones, tones, word2ph = clean_text(text, language)
                    f.write(
                        "{}|{}|{}|{}|{}|{}|{}\n".format(
                            utt,
                            spk,
                            language,
                            norm_text,
                            " ".join(phones),
                            " ".join([str(i) for i in tones]),
                            " ".join([str(i) for i in word2ph]),
                        )
                    )
                except:
                    print("err!", utt)

if 2 in stage:
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(transcription_path + ".cleaned", encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk_utt_map[spk].append(line)
            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]
    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]
        
    # TODO: 测试代码，需要删除
    train_list = val_list

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

if 3 in stage:
    assert 2 in stage
    config = json.load(open(os.path.join(config_dir, "config_template.json"), encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    config['data']['training_files'] = train_path
    config['data']['validation_files'] = val_path
    with open(os.path.join(config_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
