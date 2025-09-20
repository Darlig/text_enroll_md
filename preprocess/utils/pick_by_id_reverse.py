#!/usr/bin/env python3

import sys

source_scp = sys.argv[1]
id_file = sys.argv[2]
out_scp = sys.argv[3]

selected_ids = set()
with open(id_file) as f_id:
    for line in f_id:
        #line_sp = line.strip().split()
        #if len(line_sp) != 2:
        #    print(line)
        utt_id = line.strip().split()[0]
        selected_ids.add(utt_id)

#print(len(selected_ids))

with open(source_scp) as f_source, open(out_scp, 'w') as f_out:
    for line in f_source:
        utt_id = line.strip().split()[0]
        if utt_id not in selected_ids:
            f_out.write(line)
        #else:
        #    print(utt_id)
        #    exit()
