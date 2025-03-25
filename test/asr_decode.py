import os
import yaml
import json
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import sys

sys.path.insert(0, os.path.abspath('.'))

from yamlinclude import YamlIncludeConstructor
from model import m_dict

import re


# test_config_file = sys.argv[1]
# speech_batch_size = int(sys.argv[2])
# cross_batch_size = int(sys.argv[3])

trained_ckpt_path = sys.argv[1]
train_config = sys.argv[2]
train_data_dir = sys.argv[3]
test_wav_scp = sys.argv[4]
asr_result = sys.argv[5]

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)

train_config  = yaml.load(open(train_config), Loader=yaml.FullLoader)




FBANK_DEFAULT_SETTING = {
    'num_mel_bins': 40, 'frame_length': 25, 'frame_shift': 10
}




def load_model():
    print("trained_ckpt_path: {}".format(trained_ckpt_path))
    trained_ckpt = torch.load(trained_ckpt_path)
    trained_ckpt = trained_ckpt['model']

    #model_arch = test_config['test_model_arch']
    model_arch = train_config['model_arch']
    
    model_config = train_config['model_config']
    model = m_dict[model_arch]

    test_model = model(**model_config)
    test_model.load_state_dict(trained_ckpt)
    test_model.to('cuda:0')
    test_model.eval()
    return test_model


def load_dict(dict_path, reverse=False):
    with open(dict_path) as f_dict:
        if not reverse:
            return {k: int(v) for k, v in [re.split(r'\s+', line.strip(), maxsplit=1) for line in f_dict]}
        else:
            return {int(v): k for k, v in [re.split(r'\s+', line.strip(), maxsplit=1) for line in f_dict]}

def ctc_greedy_decode(predicted_ids, blank_id=0):
    decoded_sequences = []
    for seq in predicted_ids.T:  # 遍历 batch 内所有序列
        # print("length of seq: {}".format(len(seq)))
        # print("seq: {}...".format(seq[:20]))
        prev_id = None
        decoded_seq = []
        for idx in seq.cpu().numpy():
            if idx != blank_id and idx != prev_id:
                decoded_seq.append(idx)
            # else:
            #     print("idx: {}, prev_id: {}".format(idx, prev_id))
            #     exit()
            prev_id = idx
        decoded_sequences.append(decoded_seq)
    return decoded_sequences


import torch
import math
from collections import defaultdict

def ctc_beam_search(log_probs, beam_width=10, blank=0):
    """
    log_probs: Tensor of shape [T, C] (time steps, classes), log-softmax output
    Returns: list of best label indices (int)
    """
    print("log_probs shape: {}".format(log_probs.shape))
    # exit()
    log_probs = log_probs[:, 0: :]
    log_probs = log_probs.squeeze(1)
    print("log_probs shape: {}".format(log_probs.shape))
    T, C = log_probs.shape
    beam = [(tuple(), 0.0)]  # (sequence, score)

    for t in range(T):
        next_beam = defaultdict(lambda: -float('inf'))

        for prefix, score in beam:
            for c in range(C):
                p = log_probs[t, c].item()
                new_prefix = prefix

                if c != blank:
                    new_prefix = prefix + (c,)

                # CTC collapse rule: don't repeat same label unless separated by blank
                if len(prefix) > 0 and c == prefix[-1] and c != blank:
                    new_prefix = prefix

                next_beam[new_prefix] = log_sum_exp(next_beam[new_prefix], score + p)

        # 保留 top beam_width 条路径
        beam = sorted(next_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]

    # 合并重复（CTC collapse）+ 移除 blank
    best_seq = beam[0][0]
    collapsed = []
    prev = None
    for i in best_seq:
        if i != prev and i != blank:
            collapsed.append(i)
        prev = i

    print("collapsed length: {}".format(len(collapsed)))
    print("collapsed: {}".format(collapsed))
    return collapsed


def log_sum_exp(a, b):
    # numerically stable log-sum-exp
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


if __name__ == '__main__':
    
    model = load_model()
    id2phone = load_dict(train_data_dir+'/phone2id.txt', reverse=True)
    with open(test_wav_scp) as f_wav_scp, open(asr_result, 'w') as f_asr_result:
        lines = f_wav_scp.readlines()
        for i, line in enumerate(lines):
        # for line in f_wav_scp:
            # if i != 2:
            #     continue
            line = line.strip()
            utt, wav = line.split()
            wav = torchaudio.load(wav)[0]
            fbank = kaldi.fbank(wav, **FBANK_DEFAULT_SETTING)
            fbank = fbank.unsqueeze(0)
            fbank = fbank.to('cuda:0')
            chunk = fbank
            chunk_len = torch.tensor([chunk.size(1)]).to('cuda:0')
            speech_input = (chunk, chunk_len)
            speech_embedding, speech_mask, phn_asr_hyp = model.evaluate_sph_emb(speech_input, return_hyp=True)
            phn_asr_hyp = phn_asr_hyp.transpose(0, 1)
            # print("shape of phn_asr_hyp: {}".format(phn_asr_hyp.shape))
            # print("phn_asr_hyp[0][0]: {}...".format(phn_asr_hyp[0][0][:20]))
            import torch.nn.functional as F
            log_probs = F.log_softmax(phn_asr_hyp, dim=-1)
            print("shape of log_probs: {}".format(log_probs.shape))
            # for i in range(5):
            #     print("log_probs[{}][0]: {}...".format(i, log_probs[i][0][:20]))
            # log_probs = log_probs[:,:,:-1] # remove punk
            log_probs[:, :, 1] = -100.0
            log_probs[:, :, 2] = -100.0
            log_probs[:, :, -1] = -100.0

            decoded_sequences = ctc_beam_search(log_probs, beam_width=10)
            # decoded_sequences = ctc_beam_search(log_probs)

            # predicted_ids = log_probs.argmax(dim=-1)
            # decoded_sequences = ctc_greedy_decode(predicted_ids)
            # decoded_sequences = decoded_sequences[0]

            # print("length of decoded_sequences: {}".format(len(decoded_sequences[0])))
            # print("decoded_sequences[0]: {}".format(decoded_sequences[0]))
            # print("decoded_sequences: {}".format(decoded_sequences))

            f_asr_result.write("{} {}\n".format(utt, ' '.join([id2phone[phn] for phn in decoded_sequences])))

            # keyword_embedding, keyword_mask = model.evaluate_kw_emb((keyword, keyword_len))
            # cross_input = (speech_embedding, speech_mask, keyword_embedding, keyword_mask)
            # det_result, _ = model.evaluate_concat_attention(cross_input)
            # det_result = det_result.view(-1).to('cpu').tolist()
            # f_asr_result.write("{}\n".format(json.dumps({utt: det_result})))
