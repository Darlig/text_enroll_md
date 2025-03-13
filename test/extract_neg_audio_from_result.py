#!/usr/bin/env python3

import sys
import torch
import os

negative_result_path = sys.argv[1]
#test_result_dir = sys.argv[2]
audio_scp = sys.argv[2]

chunk_shift = 10
chunk_size = 98
target_keyword = '算了'
threshold = 0.9994250535964966

negative_result = torch.load(negative_result_path)
neg_hyp = negative_result['neg_hyp']

test_result_dir = os.path.dirname(negative_result_path)
meta_path = os.path.join(test_result_dir, 'fenyinta_meta.pt')
assert os.path.exists(meta_path)
test_material = torch.load(meta_path)
keyword2phnid = test_material['keyword2phnid']
keyword2idx = {}
for i, (keyword, phnid) in enumerate(keyword2phnid.items()):
    keyword2idx[keyword] = i

audio_path_dict = {}
with open(audio_scp) as f_scp:
    for line in f_scp:
        utt_id, audio_path = line.strip().split()
        audio_path_dict[utt_id] = audio_path

for utt_id, utt_det_result in neg_hyp.items():
    print("utt_id: {}".format(utt_id))
    audio_path = audio_path_dict[utt_id]
    print("audio_path: {}".format(audio_path))
    for i, chunk_det_result in enumerate(utt_det_result):
        #print("chunk_id: {}, chunk_det_result: {}".format(i, chunk_det_result))
        chunk_target_keyword_score = chunk_det_result[keyword2idx[target_keyword]]
        #print("chunk_id: {}, chunk target keyword score: {}".format(i, chunk_target_keyword_score))
        if chunk_target_keyword_score > threshold:
            print("chunk_id: {}, duration: {:.3f} - {:.3f} s, chunk target keyword score: {}".format(i, i * chunk_shift * 10 / 1000, (i * chunk_shift + chunk_size )* 10 / 1000, chunk_target_keyword_score))
        #if i >= 3:
        #    exit()


