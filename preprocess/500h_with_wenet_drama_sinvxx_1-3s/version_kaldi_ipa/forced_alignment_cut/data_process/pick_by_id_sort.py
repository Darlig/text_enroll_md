#!/usr/bin/env python3

import sys
import re

source_scp = sys.argv[1]
id_file = sys.argv[2]
out_scp = sys.argv[3]

selected_ids = []
with open(id_file) as f_id:
    for line in f_id:
        #line_sp = line.strip().split()
        #if len(line_sp) != 2:
        #    print(line)
        utt_id = line.strip().split()[0]
        selected_ids.append(utt_id)

#print(len(selected_ids))

source_dict = {}
with open(source_scp) as f_source:
    for line in f_source:
        utt_id, content = re.split(r'\s+', line.strip(), maxsplit=1)
        source_dict[utt_id] = content

with open(out_scp, 'w') as f_out:
    for utt_id in selected_ids:
        if utt_id in source_dict:
            f_out.write("{} {}\n".format(utt_id, source_dict[utt_id]))
        #else:
        #    print(utt_id)
        #    exit()
