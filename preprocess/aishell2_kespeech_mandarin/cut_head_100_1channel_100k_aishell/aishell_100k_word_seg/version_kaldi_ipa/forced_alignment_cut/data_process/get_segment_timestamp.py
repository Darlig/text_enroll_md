#!/usr/bin/env python3

import sys
import re
import json
import ast

source_datalist = sys.argv[1]
character_timestamp = sys.argv[2]
segment_timestamp = sys.argv[3]
# cut_datalist = sys.argv[4]

#1001501_26b0ce87 [[270, 450], [450, 490], [490, 590], [590, 650], [650, 730], [730, 770], [770, 830], [830, 890], [890, 930], [930, 990], [990, 1030], [1030, 1150], [1150, 1465]]
#{"key": "Y0000003589_8WOJb2iiULs_S00222", "sph": "/work104/weiyang/data/wenetspeech/dataset/drama_samp500h/audio_cut3/audio/train/youtube_wav/B00014/Y0000003589_8WOJb2iiULs/Y0000003589_8WOJb2iiULs_S00222.wav", "bpe_label": [815, 47, 484, 13, 14, 1566, 754, 22, 1732, 1708, 765], "phn_label": [[64, 93], [58, 143], [32, 88], [20, 21], [22, 19], [7, 184], [10, 113], [3, 61], [44, 91], [72, 52], [64, 93]], "segment_label": [[[64, 93], [58, 143], [32, 88]], [[20, 21], [22, 19]], [[7, 184], [10, 113]], [[3, 61], [44, 91]], [[72, 52], [64, 93]]], "kw_candidate": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b_kw_candidate": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

datalist_dict = {}
with open(source_datalist) as f_datalist:
    for line in f_datalist:
        utt_datalist_dict = json.loads(line.strip())
        uttid = utt_datalist_dict["key"]
        datalist_dict[uttid] = utt_datalist_dict

c_timestamp_dict = {}
with open(character_timestamp) as f_timestamp:
    for line in f_timestamp:
        uttid, content = re.split(r'\s+', line.strip(), maxsplit=1)
        c_timestamp = ast.literal_eval(content)
        c_timestamp_dict[uttid] = c_timestamp

def get_segment_timestamp(datalist_dict, c_timestamp_dict):
    seg_timestamp_dict = {}
    for uttid, utt_datalist_dict in datalist_dict.items():
        if uttid in c_timestamp_dict:
            c_timestamp = c_timestamp_dict[uttid]
            segment_label = utt_datalist_dict["segment_label"]
            # bpe_label = utt_datalist_dict["bpe_label"]
            # phn_label = utt_datalist_dict["phn_label"]
            # kw_candidate = utt_datalist_dict["kw_candidate"]
            # b_kw_candidate = utt_datalist_dict["b_kw_candidate"]
            seg_len_list = [ len(seg) for seg in segment_label ]
            seg_timestamp_list = []
            # seg_bpe_label_list = []
            # seg_phn_label_list = []
            # seg_kw_candidate_list = []
            # seg_b_kw_candidate_list = []
            i = 0
            for sl in seg_len_list:
                seg_ts = c_timestamp[i:i+sl]
                seg_timestamp_list.append(seg_ts)
                # seg_bpe_label = bpe_label[i:i+sl]
                # seg_phn_label = phn_label[i:i+sl]
                # seg_kw_candidate = kw_candidate[i:i+sl]
                # seg_b_kw_candidate = b_kw_candidate[i:i+sl]
                # seg_utt_datalist = json.dumps({
                #     "key": "{}.{}".format(uttid, i),
                #     "sph": 
                #         i += sl
            seg_timestamp_dict[uttid] = seg_timestamp_list
    return seg_timestamp_dict

with open(segment_timestamp, 'w') as f_seg_timestamp:
    seg_timestamp_dict = get_segment_timestamp(datalist_dict, c_timestamp_dict)
    for uttid, seg_timestamp_list in seg_timestamp_dict.items():
        f_seg_timestamp.write("{} {}\n".format(uttid, seg_timestamp_list))




