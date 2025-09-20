#!/usr/bin/env python3

import sys
import re
import json

cut_json = sys.argv[1]
bad_id_path = sys.argv[2]

duration_thresh = 0.4

with open(cut_json) as f_cut:
    cut_dict = json.load(f_cut)

bad_seg_ids = []
for uttid in cut_dict:
    utt_seg_ids = []
    is_bad_utt = False
    for seg in cut_dict[uttid]:
        #print(seg.keys())
        filename = seg["filename"]
        seg_id = re.sub("\.wav$", "", filename)
        utt_seg_ids.append(seg_id)
        duration = seg["extended_duration_sec"]
        if duration < duration_thresh:
            is_bad_utt = True
    if is_bad_utt:
        bad_seg_ids.extend(utt_seg_ids)

with open(bad_id_path, 'w') as f_bad_id:
    for seg_id in bad_seg_ids:
        f_bad_id.write("{}\n".format(seg_id))


