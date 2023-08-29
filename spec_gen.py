import logging
logging.basicConfig(level=logging.WARNING)
from multiprocessing import Pool

import torch
from tqdm import tqdm
import modules.commons as commons
import modules.utils as utils

from text import get_bert

config_path = 'configs/config_template.json'
hps = utils.get_hparams_from_file(config_path)

def process_line(line):
    _id, spk, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    wav_path = f'{_id}'

    bert_path = wav_path.replace(".wav", ".bert.pt")
    try:
        bert = torch.load(bert_path)
        assert bert.shape[-1] == len(phone)
    except:
        bert = get_bert(text, word2ph, language_str)
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_path)

with open(hps.data.training_files, encoding='utf-8') as f:
    lines = f.readlines()
with open(hps.data.validation_files, encoding='utf-8') as f:
    lines = lines + f.readlines()

# TODO: 目前是单线程，可以改成多线程，不过多线程还没测试
for line in tqdm(lines):
    process_line(line)

# if __name__ == '__main__':
#     with Pool(processes=4) as pool: #A100 suitable config,if coom,please decrease the processess number.
#         for _ in tqdm(pool.imap_unordered(process_line, lines)):
#             pass
