import os
import yaml
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import sys

sys.path.insert(0, os.path.abspath('.'))

from yamlinclude import YamlIncludeConstructor
from model import m_dict

import re

import math
from collections import defaultdict

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
            prev_id = idx
        decoded_sequences.append(decoded_seq)
    return decoded_sequences


def beam_search_ctc(probabilities, beam_width=10, blank_id=0, special_tokens=None):
    """
    使用 Beam Search 对 CTC 概率进行解码，返回最佳输出序列（音素序列）。
    
    参数:
      probabilities: numpy 数组，形状为 (T, V)，T 为时间步数，V 为词汇表大小，
                     表示每个时间步各个标签的概率分布。
      beam_width: Beam search 的宽度，默认值为 10。
      blank_id: 表示 CTC 空白符号的索引，默认值为 0。
      
    返回:
      final_sequence: 解码得到的最佳音素序列（以标签索引表示），
                      序列中连续重复的标签已合并，并去除了空白符号。
    """
    if special_tokens is None:
        special_tokens = set()

    T, V = probabilities.shape
    # 初始 beam，键为当前序列（元组），值为 (p_blank, p_non_blank)：
    # p_blank 表示以空白结束的概率；p_non_blank 表示以非空白结束的概率。
    beam = {(): (1.0, 0.0)}
    
    # 对每个时间步进行扩展
    for t in range(T):
        new_beam = {}
        for prefix, (p_b, p_nb) in beam.items():
            # 当前前缀的总概率
            prob_total = p_b + p_nb

            # 1. 考虑空白符扩展：空白不改变前缀
            new_prob = prob_total * probabilities[t, blank_id]
            if prefix in new_beam:
                prev_p_b, prev_p_nb = new_beam[prefix]
                new_beam[prefix] = (prev_p_b + new_prob, prev_p_nb)
            else:
                new_beam[prefix] = (new_prob, 0.0)
            
            # 2. 对每个非空白符进行扩展
            for v in range(V):
                if v == blank_id:
                    continue

                if v in special_tokens:
                    # 特殊 token 类似于 blank：不扩展路径，但应该更新原路径的得分
                    new_prob = prob_total * probabilities[t, v]
                    if prefix in new_beam:
                        prev_p_b, prev_p_nb = new_beam[prefix]
                        new_beam[prefix] = (prev_p_b + new_prob, prev_p_nb)
                    else:
                        new_beam[prefix] = (new_prob, 0.0)
                    continue

                # 处理普通 token：扩展路径
                prob_v = probabilities[t, v]
                new_prefix = prefix + (v,)
                # 如果当前 token 与上一个 token 相同，仅允许从 p_blank 扩展（防止重复计分）
                if prefix and v == prefix[-1]:
                    p_new = p_b * prob_v
                else:
                    p_new = prob_total * prob_v
                
                if new_prefix in new_beam:
                    prev_p_b, prev_p_nb = new_beam[new_prefix]
                    new_beam[new_prefix] = (prev_p_b, prev_p_nb + p_new)
                else:
                    new_beam[new_prefix] = (0.0, p_new)
        
        # 保留 beam_width 个得分最高的候选序列
        sorted_beam = sorted(new_beam.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)
        beam = {prefix: scores for prefix, scores in sorted_beam[:beam_width]}
    
    # 从最终 beam 中选出总概率最高的序列
    best_prefix, (p_b, p_nb) = max(beam.items(), key=lambda x: x[1][0] + x[1][1])
    
    # 对输出序列进行后处理：去除空白符，并合并重复的标签
    final_sequence = []
    previous = None
    for token in best_prefix:
        if token != blank_id and token != previous and token not in special_tokens:
            final_sequence.append(token)
        previous = token
    return final_sequence


def ctc_beam_search(log_probs, beam_width=10, blank=0, special_tokens=None):
    """
    log_probs: Tensor of shape [T, C] (time steps, classes), log-softmax output
    Returns: list of best label indices (int)
    """
    if special_tokens is None:
        special_tokens = set()

    T, C = log_probs.shape
    beam = [(tuple(), 0.0)]  # (sequence, score)

    for t in range(T):
        next_beam = defaultdict(lambda: -float('inf'))

        for prefix, score in beam:
            for c in range(C):
                p = log_probs[t, c].item()
                new_prefix = prefix

                if c != blank and c not in special_tokens:
                    new_prefix = prefix + (c,)

                # CTC collapse rule: don't repeat same label unless separated by blank
                if len(prefix) > 0 and c == prefix[-1] and c != blank and c not in special_tokens:
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
            probs = torch.exp(log_probs)

            punk = log_probs.size(2) - 1
            assert log_probs.size(1) == 1
            log_probs = log_probs.squeeze(1)

            assert probs.size(1) == 1
            probs = probs.squeeze(1)

            decoded_sequences = beam_search_ctc(probs, beam_width=10, special_tokens={1, 2, punk})

            # decoded_sequences = ctc_beam_search(log_probs, beam_width=5, special_tokens={1, 2, punk})

            # predicted_ids = log_probs.argmax(dim=-1)
            # decoded_sequences = ctc_greedy_decode(predicted_ids)
            # decoded_sequences = decoded_sequences[0]

            f_asr_result.write("{} {}\n".format(utt, ' '.join([id2phone[phn] for phn in decoded_sequences])))
