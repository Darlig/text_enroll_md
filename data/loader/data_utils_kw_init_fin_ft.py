# ref: wenet processor.py
import re
import math
import random
import torch
import torchaudio
import json
import copy
import itertools

import numpy as np

from torch.nn.utils.rnn import pad_sequence
from scipy.io import wavfile
from scipy import signal
from typing_extensions import List, Dict, Tuple, Iterator, Any, Optional


RE_PATTERN = {'space': re.compile(r" +"), 'dot': re.compile(r"\.")}

# random factory
RANDOM_FACTOR = {
    'beta':np.random.beta, 'uniform': np.random.uniform, 'int': np.random.randint, 'random': np.random.random
}

# Pre-Defined Special Token
TEXT_SPEC_TOKEN = {
    'sos': None, 'eos': None, 'sok': None, 'eok': None, 'unk': None, 'with_trans': None,
    'psok': None, 'peok': None, 'punk': None
}


#TODO: Max mix speech
def permuate_labels(labels: List[int], conject_token: List=None, max_mix_num: int=3):
    ids = [x for x in range(len(labels))]
    permuates = []
    max_n_permuate = math.factorial(max_mix_num)
    for one_idx in itertools.permutations(ids):
        if conject_token:
            one_per = [labels[x] + [conject_token] for x in one_idx]
            one_per[-1] = one_per[-1][:-1]
        else:
            one_per = [labels[x] for x in one_idx]
        permuates.append(one_per)

    if len(permuates) < max_n_permuate: # assume 3mix in data
        pad_per = permuates[0]
        for x in range(max_n_permuate-len(permuates)):
            permuates.append(pad_per)
    return permuates


# split str "0.1 0.2 0.3" to float [0.1 0.2 0.3] 
# str to int; int to sym; tensor to str
def sym2float(sym_list: List[int]):
    int_list = list(map(lambda x: float(x), sym_list.split(" ")))
    return int_list
def int2sym(int_list: List[int]):
    if not isinstance(int_list, list):
        int_list = [int_list]
    int_list = [str(x) for x in int_list]
    return int_list
def sym2int(sym_list: List[int]):
    int_list = list(map(lambda x: int(x), sym_list.split(" ")))
    return int_list
def tensor2str(t: torch.Tensor):
    if isinstance(t, torch.Tensor):
        t = t.numpy()
    t = list(t)
    t = list(map(lambda x: str(x), t))
    return t



# save wav as PCM_S 16bit 16k: always use to test code
def save_wav(wav: torch.Tensor, names: str):
    if isinstance(names, list):
        names = "_".join(names)
    torchaudio.save(
        "{}.wav".format(names), wav, sample_rate=16000, encoding="PCM_S", bits_per_sample=16
    )

# Splice feats: append context 
def splice_feats(feats: torch.Tensor, left_context: int, right_context: int, seq=False) -> torch.Tensor:
    frames, nmel = feats.size()
    l_padding = torch.ones_like(torch.rand(left_context, nmel))
    r_padding = torch.ones_like(torch.rand(right_context, nmel))
    l_padding *= feats[0]
    r_padding *= feats[-1]
    feats = torch.cat([l_padding, feats, r_padding], dim=0)
    if seq:
        return feats
    else:
        splice_v = []
        for i in range(left_context + right_context + 1):
            v = feats[i:frames + i]
            splice_v.append(v)
        feats = torch.cat([v for v in splice_v], dim=-1)
        return feats

# max_energy: find the max energy frames: TODO: partially duplicated with time shifting 
def max_energy(wav: torch.Tensor, frame_length: int=400, hop_length: int=160, shift_type: str='mid') -> int:
    wav = wav.view(-1) # assume wav is singal channel speech [1, num_samples] multi channel is not supportTODO:
    num_frames = 1 + (len(wav) - frame_length) // hop_length
    energy = torch.zeros(num_frames, dtype=torch.float32)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = wav[start:end]
        energy[i] = torch.sum(frame ** 2)
    # max frame energy
    idx_frame = torch.argmax(energy) 
    dice_frame = random.randint(0,5)
    if dice_frame % 2 == 0:
        idx_frame = idx_frame - dice_frame
        idx_frame = idx_frame if idx_frame > 0 else 0
    else:
        idx_frame = idx_frame + dice_frame
    #idx_frame = idx_frame - dice_frame
    idx_sample = idx_frame * hop_length + frame_length 
    return idx_sample

# flatten list [[1,2,3],[4,5,6,[7]]] => [1,2,3,4,5,6,7]
def unfold_list(lst: List[int]) -> List[int]:
    if not isinstance(lst, list):
        lst = [lst]
    l = _unfold_list(lst)
    trans = int if re.search(RE_PATTERN['dot'], l) == None else float
    l = [trans(i) for i in l.split(" ") if i !=""]
    return l
# sub method of unfold_list 
def _unfold_list(lst: List[int]) -> List[int]:
    new = ""
    for x in lst:
        if isinstance(x, list):
            x = _unfold_list(x)
        x = str(x)
        new = new + x + " "
    return new

#concat tensor
def concat_tensor(
    data_list: List[torch.Tensor],
    seq_padding: bool=False,
    padding_value: int=0
) -> torch.Tensor:
    if seq_padding:
        tensor = pad_sequence(data_list, batch_first=True, padding_value=padding_value)
    else:
        tensor = torch.cat([x.unsqueeze(0) for x in data_list], dim=0)
    return tensor


# random one sample from a pool
def random_one(pools: List[int], pool_len: int) -> Tuple[List]:
    return (pools[random.randint(0, pool_len-1)])


# spec augmentation
def spec_augment(
        spec: torch.Tensor, config: Dict
    ) -> torch.Tensor:
    assert isinstance(spec, torch.Tensor)
    num_t_mask = config.get('num_t_mask', 2)
    num_f_mask = config.get('num_f_mask', 2)
    max_t = config.get('max_t', 20)
    max_f = config.get('max_f', 10)
    spec_prob = config.get('spec_prob', 0.5)
    aug_spec = spec.clone().detach()
    if random.uniform(0, 1) > spec_prob:
        return aug_spec
    else:
        max_frames = aug_spec.size(0)
        max_freq = aug_spec.size(1)
        # print("max_frames: {}, max_freq: {}".format(max_frames, max_freq))
        # time mask
        for i in range(num_t_mask):
            start = np.random.randint(0, max_frames - 1)
            length = np.random.randint(1, max_t)
            end = min(max_frames, start + length)
            aug_spec[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = np.random.randint(0, max_freq - 1)
            length = np.random.randint(1, max_f)
            end = min(max_freq, start + length)
            aug_spec[:, start:end] = 0
            # print("start: {}, end: {}".format(start, end))
        return aug_spec

# Speech augmentation: reverb, change speed
def wav_augment(
        waveform: torch.Tensor, config: Dict, sample_rate: int=16000
    ) -> torch.Tensor:
    # add reverb and change speech 
    # TODO: if change speech alignment and segment should change too!!!
    if config.get("volume", False):
        volume_sampler = config['volume']['sampler']
        volume_sampler_config = config['volume']['config']
        ratio = RANDOM_FACTOR[volume_sampler](**volume_sampler_config)
        waveform = ratio * waveform
    
    if config.get('speed_perturb', False):
        speed_factor = config.get('speed_perturb').get('factors', [0.9, 1.0, 1.1])
        factor = random.choice(speed_factor)
        if factor != 1.0:
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, [['tempo', str(factor)], ['rate', str(sample_rate)]]
            )
    return waveform


def reverb_aug(waveform: torch.Tensor, config: Dict, rirs: str=None) -> torch.Tensor:
    if len(waveform.size()) == 2:
        n = waveform.size(0)
    else:
        n = waveform.dim()
    assert(n == 1)
    if rirs:
        rirs_prob = config.get('rirs_prob', 0.4)
        if random.uniform(0,1) < rirs_prob:
            if isinstance(rirs, list):
                rirs_file = random.choice(rirs)
            _, rirs_src = wavfile.read(rirs_file)
            rirs_src = rirs_src / np.sqrt(np.sum(rirs_src**2))
            rirs_src = rirs_src.astype(np.float32)
            l = waveform.size(1)
            waveform = waveform[0].numpy()
            waveform = signal.convolve(waveform, rirs_src)[:l]
            waveform = torch.from_numpy(waveform)
            waveform = waveform.view(1,-1)
    return waveform

def add_noise(speech: torch.Tensor, noise: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    snr = random.randint(3, 10)
    signal_power = (speech**2).mean()
    noise_power = (noise**2).mean()

    target_noise_power = signal_power / (10**(snr / 10))

    scaling_factor = torch.sqrt(target_noise_power / noise_power)
    adjusted_noise = noise * scaling_factor
    return (speech + adjusted_noise, adjusted_noise)


def make_mix_wav(
        speech: torch.Tensor, ratios: List[float], noise: Optional[torch.Tensor] = None, 
        noise_ratio: Optional[List] = None, delay_prob: float=0.35, delay_type='random'
    )-> torch.Tensor:
    assert (delay_type in ['random', 'sequential'])
    if (random.random() < delay_prob) and (len(speech) > 1):
        for delay_idx in range(1, len(speech)): # apply delay from the second wavform
            if delay_idx == 1:
                delay_len = random.randint(0, 16000*3)
            else:
                if delay_type == 'sequential':
                    delay_len = delay_len + random.randint(2000, 16000*3)
                else:
                    delay_len = random.randint(0, 16000*3)
            zero_padding = torch.zeros(1, delay_len)
            speech[delay_idx] = torch.cat([zero_padding, speech[delay_idx]], dim=1)

    max_len = max([s.size(1) for s in speech])
    speech = [padding_wav(s, max_len, 'zero') for s in speech] # padding_wav(target_wav, padding_len, padding_type)
    noise = [padding_wav(n, max_len, 'repeat') for n in noise]

    rms = [math.sqrt((s**2).mean()) for s in speech]
    rms = list(map(lambda x: x if x > 0.0001 else 1, rms)) # avoid devide very small value 0
    max_rms = max(rms)

    scaled_wav = [speech[i]*(max_rms/rms[i]) for i in range(len(speech))]
    scaled_wav = [scaled_wav[i]*ratios[i] for i in range(len(scaled_wav))]

    mix_wav = sum(scaled_wav)

    if len(noise) > 0:
        noise = sum(noise)
        mix_wav, noise = add_noise(mix_wav, noise) # Signal power is temporary set to SNR=1~10dB
        wav_group = scaled_wav + [noise]
    else:
        wav_group = scaled_wav[:]
    mix_wav = torch.clamp(mix_wav, -1.0, 1.0)

    return [mix_wav] + wav_group


# padding wav with zeros
def padding_wav(waveform: torch.Tensor, length: int, padding_type: str='zero')->torch.Tensor:
    assert (padding_type in ['zero', 'repeat'])
    assert (waveform.dim() == 2)
    assert (waveform.size(0) == 1)
    wav_len = waveform.size(1)
    if padding_type == 'repeat':
        if wav_len < length:
            repeat_time = length // wav_len + 1
            waveform = torch.tile(waveform.view(-1), (repeat_time,))
            waveform = waveform.view(1, -1)
        waveform = waveform[:,:length]
    else:
        assert (waveform.size(1) <= length)
        zeros = torch.zeros(1, length - wav_len)
        waveform = torch.cat([waveform, zeros], dim=1)
    return waveform

# trim wav by lenth and segment
def trim_wav(wav: torch.Tensor, segment: List[int]) -> torch.Tensor:
    head, tail = segment
    if head == -1:
        # when head = -1 tail is the target windows length
        head = 0
        wav_size = wav.size(1)
        res = wav_size - tail
        head = 0 if res <= 0 else int(random.uniform(0, res))
        tail = head + tail
    wav = wav[:,head: tail]
    return wav


# get segment idx by time stamp
def got_seg(v, segment):
    diff = float('inf')
    nearest_value, nearest_index = None, None
    for i, value in enumerate(segment):
        if abs(value - v) < diff:
            diff = abs(value - v)
            nearest_value = value
            nearest_index = i
    return nearest_value, nearest_index


# make corruption pairs, maybe triple or more
# detach_f: detach_functions, for self corruption the datalist will format as json object so detach_f will be json.loads
# for none target corruption detach_f is lambda x: x
def make_corrupt_party(
        corrupt_list: List[Any], corrupt_list_len: int, config: Dict, corruption_type: str='self'
    )-> Tuple[List, Dict, int]:
    assert (corruption_type in ['self', 'noise'])
    ratios = list()
    corruption_material = dict()
    n = config.get('num_corrupt')
    max_corrupt_num = config.get('num_corrupt')
    
    if config.get('random_num', False):
        assert(n > 1)
        n = RANDOM_FACTOR['int'](1, n+1)
    if config.get('prob', 1.2) < 1:
        dice = random.uniform(0,1)
        if dice > config.get('prob'):
            n = 0

    sampler = config.get('sampler')
    assert (sampler in RANDOM_FACTOR.keys())
    sampler_config = config.get('sampler_config')
    if corruption_type == 'self':
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)
    
    for i in (range(n)):
        one_corrupt = corrupt_list[np.random.randint(0, corrupt_list_len)]
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)
        if isinstance(one_corrupt, dict):
            corruption_material[i+1] = one_corrupt
        elif isinstance(one_corrupt, str):
            corruption_material[i+1] = {'sph': one_corrupt}
        else:
            raise NotImplementedError("corrupt list error")
    return (ratios, corruption_material, n, max_corrupt_num)


# time shiftting: shiftting keywords from waveforme to make sure that: when mix tow keywords they will completely overlaped
def time_shifting(
        wav: torch.Tensor, frame_length: int=400, hop_length: int= 160, shift_type: str='mid'
    )-> torch.Tensor:
    wav = wav.view(-1) # assume wav is singal channel speech [1, num_samples] multi channel is not supportTODO:
    num_frames = 1 + (len(wav) - frame_length) // hop_length
    energy = torch.zeros(num_frames, dtype=torch.float32)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = wav[start:end]
        energy[i] = torch.sum(frame ** 2)
    # max frame energy
    idx_frame = torch.argmax(energy) 
    idx_sample = idx_frame * hop_length + frame_length 
    if shift_type == 'mid':
        dest_pos = int(wav.size(0) / 2)
    shift = int(dest_pos-idx_sample)
    wav = torch.roll(wav, shift)
    return wav.view(1, -1) # NOTE: convert back to singal channel [1, num_sample]


# make segment: make segment head and tail => [0.5, 1.5] mains: wav will trimed from the 0.5s to 1.5s (1s) 
def make_segment(
        segments: List[int], win_len: int, trim_type: str='raw', dither: bool=False, sample_rate: int=16000
    ) -> Tuple[int]:
    seg_head_tail = []
    idx_head_tail = []
    for seg in segments:
        if seg == -1:
            seg_head_tail.append([-1, int(win_len*sample_rate)])
            idx_head_tail.append([-1, -1])
            continue
        if trim_type == 'raw':
            seg_head = seg[0]
            if dither:
                seg_head = seg_head - random.uniform(0, 0.1)
                seg_head = seg_head if seg_head > 0 else 0
            seg_tail = seg_head + win_len
            seg_head_tail.append([int(seg_head*sample_rate), int(seg_tail*sample_rate)])
            idx_head_tail.append([0, len(seg)])
        elif trim_type == 'completetrim': 
            # cut setence from one utterance; the sub-utterance will contain a complete content
            leng_wav = seg[-1][-1]
            if leng_wav > win_len:
                seg_head_range = leng_wav - win_len
                _, seg_idx = got_seg(seg_head_range, [s[0] for s in seg])
                seg_head_idx = seg_idx if seg_idx <= 1 else np.random.randint(0, seg_idx-1)
                seg_head = seg[seg_head_idx][0]
                seg_tail = seg_head + win_len
                _, seg_tail_idx = got_seg(seg_tail, [s[1] for s in seg])
                seg_tail_idx = seg_tail_idx + 1 if seg_tail_idx < len(seg) - 1 else seg_tail_idx
                seg_tail = seg[seg_tail_idx][1]
            else:
                seg_head = 0
                seg_tail = seg[-1][-1]
                seg_head_idx = 0 
                seg_tail_idx = len(seg)
            seg_head_tail.append([int(seg_head*sample_rate), int(seg_tail*sample_rate)])
            idx_head_tail.append([seg_head_idx, seg_tail_idx])
        else:
            raise NotImplementedError("Only support raw")
    return (seg_head_tail, idx_head_tail)


# detach corruption
def detach_corruption(material: Dict) -> Tuple[List, List, List, List, List, List]:
    keywords = [] 
    phn_labels = []
    bpe_labels = []
    labels = []
    kw_candidates = []
    b_kw_candidates = []
    for i, (_idx, info) in enumerate(material.items()):
        if 'keyword' in info:
            keywords.append(info['keyword'])
        if 'label' in info:
            labels.append(info['label'])
        if 'phn_label' in info:
            phn_labels.append(info['phn_label'])
        if 'bpe_label' in info:
            bpe_labels.append(info['bpe_label'])
        if 'b_kw_candidate' in info:
            b_kw_candidates.append(info['b_kw_candidate'])
        if 'kw_candidate' in info:
            kw_candidates.append(info['kw_candidate'])
    return keywords, labels, phn_labels, bpe_labels, kw_candidates, b_kw_candidates

# insert special token in label sequence such as SOS: 0(start of sentence) 
# 1 2 3 4 5 -> "0" 1 2 3 4 5
def inject_special_token(
        keyword: List[int], keyword_length: int, label: List=None, 
        positive: bool=True, keyword_pos: int=None, special_token: Dict={}, bpe_label: List=None, 
        bpe_candidate: List=None
    )->Tuple[List, List, List, int]:
    TEXT_SPEC_TOKEN.update(special_token)
    new_phn_label = copy.deepcopy(label)
    new_bpe_label = copy.deepcopy(bpe_label) if bpe_label else [0]
    new_keyword = copy.deepcopy(keyword)

    if TEXT_SPEC_TOKEN['sos'] != None: # start of sentence
        new_phn_label = [TEXT_SPEC_TOKEN['sos']] + new_phn_label
        keyword_pos = keyword_pos + 1  if keyword_pos != None else keyword_pos # one token insert before the keyword
        
    if TEXT_SPEC_TOKEN['eos'] != None: # end of sentence
        new_phn_label = new_phn_label + [TEXT_SPEC_TOKEN['eos']] 

    if TEXT_SPEC_TOKEN['psok'] != None: # start of keyword
        new_keyword.insert(0, [TEXT_SPEC_TOKEN['psok']])

    if TEXT_SPEC_TOKEN['peok'] != None: # end of keyword
        new_keyword.insert(len(new_keyword), [TEXT_SPEC_TOKEN['peok']])

    if (TEXT_SPEC_TOKEN['with_trans']) and (positive): # modify keyword in label
        new_phn_label[keyword_pos: keyword_pos+keyword_length] = new_keyword
        if bpe_label:
            new_bpe_label.insert(0, [TEXT_SPEC_TOKEN['sok']])
            new_bpe_label.insert(len(new_keyword), [TEXT_SPEC_TOKEN['eok']])

    return (new_keyword, new_phn_label, new_bpe_label, keyword_pos)

# snipe_edges for waveform
def snipe_edge(waveform: torch.Tensor, hop_length: int=160):
    num_samples = waveform.size(1)
    edges = num_samples % hop_length
    return waveform[:,0:num_samples-edges]

def make_keyword(
        candidate_seq: List[Any], kw_lexicon: List[Any],
        finetune_data: int=None
    ) -> Tuple[List, int, int, bool, int]:

    assert finetune_data in [0, 1]
    if finetune_data == 0:
        keyword = candidate_seq
        keyword_pos = 0
        pos = True
        target = torch.tensor([1])
    else:
        keyword = random.choice(kw_lexicon)
        keyword_pos = -1
        pos = False
        target = torch.tensor([0])
    return (keyword, keyword_pos, len(keyword), pos, target)

