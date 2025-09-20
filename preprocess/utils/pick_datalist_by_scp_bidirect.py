#!/usr/bin/env python3

import sys
import re
import json

source_datalist_path = sys.argv[1]
id_scp_path = sys.argv[2]
out_datalist_path = sys.argv[3]
out_scp_path = sys.argv[4]

datalist_dict = {}
with open(source_datalist_path) as f_datalist:
    for line in f_datalist:
        utt_datalist_dict = json.loads(line.strip())
        datalist_dict[utt_datalist_dict["key"]] = utt_datalist_dict

scp_dict = {}
with open(id_scp_path) as f_scp:
    for line in f_scp:
        uttid, content = re.split(r'\s+', line.strip(), maxsplit=1)
        scp_dict[uttid] = content

with open(out_datalist_path, 'w') as f_out_datalist, open(out_scp_path, 'w') as f_out_scp:
    for uttid in scp_dict:
        if uttid in datalist_dict:
            f_out_datalist.write("{}\n".format(json.dumps(datalist_dict[uttid])))
            f_out_scp.write("{} {}\n".format(uttid, scp_dict[uttid]))
