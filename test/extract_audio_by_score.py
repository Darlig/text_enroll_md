import numpy as np
import sys
import json
import os

in_score = sys.argv[1]
in_scp = sys.argv[2]
# out_png = sys.argv[3]
keywords = [5, 6]
threshold = 0.1

scp_dict = {}
with open(in_scp) as f_scp:
    for line in f_scp.readlines():
        utt, path = line.strip().split()
        scp_dict[utt] = path

for keyword in keywords:
    out_dir = "score_less_{}/{}".format(threshold, keyword)
    os.makedirs(out_dir, exist_ok=True)
    scores = []
    with open(in_score) as f:
        for line in f.readlines():
            line_dict = json.loads(line.strip())
            key = line_dict['key']
            if line_dict['target_keyword'] == keyword:
                utt_score = [ chunk_score[keyword] for chunk_score in line_dict['scores'] ]
                utt_score = max(utt_score)
                if utt_score < threshold:
                    os.system("cp {} {}/".format(scp_dict[key], out_dir))
                # print(len(line_dict['scores'][0]))
                # scores.append(utt_score)





