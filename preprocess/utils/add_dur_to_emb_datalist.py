#!/usr/bin/env python3

import sys
import json

source_datalist = sys.argv[1]
utt2dur = sys.argv[2]
output_datalist = sys.argv[3]

utt2dur_dict = {}
with open(utt2dur, 'r') as f_dur:
    for line in f_dur:
        uttid, dur = line.strip().split()
        utt2dur_dict[uttid] = dur

with open(source_datalist, 'r') as f_source_data, open(output_datalist, 'w') as f_output_data:
    for line in f_source_data:
        utt_dict = json.loads(line.strip())
        uttid = utt_dict['key']
        utt_dict['duration'] = float(utt2dur_dict[uttid])
        f_output_data.write(f"{json.dumps(utt_dict)}\n")
        
