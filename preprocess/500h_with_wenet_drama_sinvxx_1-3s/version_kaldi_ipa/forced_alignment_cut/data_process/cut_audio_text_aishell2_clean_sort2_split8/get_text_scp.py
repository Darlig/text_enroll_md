#!/usr/bin/env python3

import sys
import json
import re

cut_json_path = sys.argv[1]
text_scp_path = sys.argv[2]

with open(cut_json_path) as f_cut:
    cut_dict = json.load(f_cut)

with open(text_scp_path, 'w') as f_text:
    for uttid in cut_dict:
        for seg in cut_dict[uttid]:
            filename = seg["filename"]
            seg_id = re.sub("\.wav$", "", filename)
            seg_text = seg["text_segment"]
            f_text.write("{} {}\n".format(seg_id, seg_text))



#{
#  "IC0001W0001": [
#    {
#      "segment_id": 0,
#      "filename": "IC0001W0001_segment_000.wav",
#      "file_path": "cut_audio_text_aishell2_clean_sort2_split8/output/sp00/IC0001W0001_segment_000.wav",
#      "word_range": [
#        0,
#        0
#      ],
#      "total_chars": 2,
#      "text_segment": "厨房",
#      "original_time_range_ms": [
#        530,
#        1010
#      ],
#      "extended_time_range_ms": [
#        430,
#        1110
#      ],
#      "original_duration_sec": 0.48,
#      "extended_duration_sec": 0.6800000000000002,
#      "extend_before_ms": 100,
#      "extend_after_ms": 100
#    },
#    {
#      "segment_id": 1,
#      "filename": "IC0001W0001_segment_001.wav",
#      "file_path": "cut_audio_text_aishell2_clean_sort2_split8/output/sp00/IC0001W0001_segment_001.wav",
#      "word_range": [
#        1,
#        1
#      ],
#      "total_chars": 2,
#      "text_segment": "用具",
#      "original_time_range_ms": [
#        1030,
#        1685
#      ],
#      "extended_time_range_ms": [
#        930,
#        1785
#      ],
#      "original_duration_sec": 0.655,
#      "extended_duration_sec": 0.8549999999999999,
#      "extend_before_ms": 100,
#      "extend_after_ms": 100
#    }
#  ],
#  "IC0001W0002": [
