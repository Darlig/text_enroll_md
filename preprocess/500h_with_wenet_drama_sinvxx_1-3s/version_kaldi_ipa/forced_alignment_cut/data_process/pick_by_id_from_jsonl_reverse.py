#!/usr/bin/env python3

import sys
import json

source_json = sys.argv[1]
id_file = sys.argv[2]
out_json = sys.argv[3]

selected_ids = set()
with open(id_file) as f_id:
    for line in f_id:
        #line_sp = line.strip().split()
        #if len(line_sp) != 2:
        #    print(line)
        utt_id = line.strip().split()[0]
        selected_ids.add(utt_id)

#print(len(selected_ids))

with open(source_json) as f_source, open(out_json, 'w') as f_out:
    for line in f_source:
        line_dict = json.loads(line.strip())
        if line_dict['key'] not in selected_ids:
            f_out.write(line)
        #utt_id = line.strip().split()[0]
        #if utt_id in selected_ids:
        #    f_out.write(line)
        ##else:
        ##    print(utt_id)
        ##    exit()
