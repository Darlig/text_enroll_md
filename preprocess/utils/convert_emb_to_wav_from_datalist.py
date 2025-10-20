#!/usr/bin/env python3

import sys
import json

source_datalist = sys.argv[1]
wav_scp = sys.argv[2]
output_datalist = sys.argv[3]

wav_path_dict = {}
with open(wav_scp, 'r') as f_scp:
    for line in f_scp:
        uttid, wav_path = line.strip().split()
        wav_path_dict[uttid] = wav_path

with open(source_datalist, 'r') as f_source_data, open(output_datalist, 'w') as f_output_data:
    for line in f_source_data:
        utt_dict = json.loads(line.strip())
        key = utt_dict['key']
        utt_wav_path = wav_path_dict[key]
        del utt_dict['sph_emb']
        utt_dict.update({'sph': utt_wav_path})
        f_output_data.write(f"{json.dumps(utt_dict)}\n")
