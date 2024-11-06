#!/usr/bin/env python3

import sys
import torch
import json

result_path = sys.argv[1]
score_path = sys.argv[2]

test_result = torch.load(result_path)
positive_result = test_result['pos_hyp']
utt2keyword_idx = test_result['pos_utt2keyword_idx']

with open(score_path, 'w') as f:
    for utt, results in positive_result.items():
        line_dict = {
            "key": utt,
            "scores": results,
            "target_keyword": utt2keyword_idx[utt]
        }
        f.write(json.dumps(line_dict) + '\n')
