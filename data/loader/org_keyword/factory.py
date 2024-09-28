# ref: wenet processor.py
import re
import random
import torch
import torchaudio
import json
import copy
import itertools

import data.loader.kaldi_io as kaldi_io
import torchaudio.compliance.kaldi as kaldi
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from scipy.io import wavfile
from scipy import signal

# spectrugram
def spectrum(wav, n_fft=512, hop_length=160):
    extractor = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
    spec = extractor(wav)
    spec = spec.squeeze(0)
    return spec.transpose(0,1).contiguous()  

def mel_padding_wav(waveform, window_size=400, window_shift=160):
    num_samples = waveform.size(0)
    reversed_waveform = torch.flip(waveform, [0])
    m = (num_samples + (window_shift // 2)) // window_shift
    pad = window_size // 2 - window_shift // 2
    pad_right = reversed_waveform[:pad]
    if pad > 0:
        # torch.nn.functional.pad returns [2,1,0,1,2] for 'reflect'
        # but we want [2, 1, 0, 0, 1, 2]
        pad_left = reversed_waveform[-pad:]
        waveform = torch.cat((pad_left, waveform, pad_right), dim=0)
    else:
        # pad is negative so we want to trim the waveform at the front
        waveform = torch.cat((waveform[-pad:], pad_right), dim=0)
    return waveform.view(1,-1)

def get_mel_scale(n_mels=80, sample_rate=16000, f_min=0, f_max=None, n_stft=201, norm=None, mel_scale='htk'):
    f_max = sample_rate // 2
    filter_bank = torchaudio.functional.melscale_fbanks(
        n_stft, f_min, f_max, n_mels, sample_rate, norm, mel_scale
    )
    return filter_bank

def permuate_labels(labels, conject_token=None):
    ids = [x for x in range(len(labels))]
    permuates = []
    for one_idx in itertools.permutations(ids):
        if conject_token:
            one_per = [labels[x] + [conject_token] for x in one_idx]
            one_per[-1] = one_per[-1][:-1]
        else:
            one_per = [labels[x] for x in one_idx]
        permuates.append(one_per)

    if len(permuates) < 6: # assume 3mix in data
        pad_per = permuates[0]
        for x in range(6-len(permuates)):
            permuates.append(pad_per)
    return permuates

def mel_spectrum(
        waveform, 
        win_length=400, 
        hop_length=160, 
        n_fft=400, 
        win_fn='hamm',
        pad_mode='reflect',
        pow=2,
        center=False, onesided=True, 
    ):
    waveform = mel_padding_wav(waveform.view(-1), win_length, hop_length)
    win_fn = torch.hamming_window
    window = win_fn(win_length)
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=False,
        onesided=onesided,
        return_complex=True,
    )
    filter_bank = get_mel_scale(n_mels=80)
    spec_f = spec_f.abs().pow(pow)
    mel_spec = torch.matmul(spec_f.transpose(-1,-2), filter_bank).transpose(-1,-2)
    return mel_spec.squeeze(0).transpose(0,1)


# Pre-Defined None-Tensor Key & CTC Tag
NONE_TENSOR_KEY = [
    'wav', 'key', 'sph', 'corruption_material', 'segment', 'segment_idx',
    'n_scorrupt', 'n_ncorrupt', 'num_corrupt', 'rirs', 'neg_candidate',
    'corrupt', 'self_corruption', 'none_target_corruption', 'nframes', 'ref1', 'ref2', 'ref3', 'ref0'
]
CTC_KEY = [
    'label', 'crpt_label', 'phn_label', 'bpe_label', 'c_phn_label', 'c_bpe_label',
    'c_bpe_label0','c_bpe_label', 'c_bpe_label1', 'c_bpe_label2',
    'c_phn_label0', 'c_phn_label1','c_phn_label2', 'neg_label', 'fifo_label',
    'per0', 'per1', 'per2', 'per3', 'per4', 'per5'

] # to be extended


# Pre-Defined Special Token
TEXT_SPEC_TOKEN = {
    'sos': None, 'eos': None, 'sok': None, 'eok': None, 'unk': None, 'with_trans': None,
    'psok': None, 'peok': None, 'punk': None
}

# input loader and feats extractor mapping
INPUT_DATA_LOADER = {
    'raw': torchaudio.load, 'kaldi': kaldi_io.read_mat, 'torch': torch.load,
    'rm_sr': lambda x: x[0], 'copy': copy.deepcopy, 'empty': lambda x: x
}

# mfcc fbank spectrum factory
FEATS_EXTRACTOR = {
    'mfcc': kaldi.mfcc, 'fbank': kaldi.fbank, 
    #'spectrum': torchaudio.functional.spectrogram, 'empty': lambda x: x 
    'spectrum': spectrum, 'mel_spec': mel_spectrum
}

# some defualt setting for feats extractor:
# mfcc, fbank, spectrum
MFCC_DEFAULT_SETTING = {
    'num_mel_bins': 23, 'num_ceps': 13, 'frame_length': 25, 'frame_shift': 10,
    'energy_floor': 0.0, 'low_freq': 20
}
FBANK_DEFAULT_SETTING = {
    'num_mel_bins': 40, 'frame_length': 25, 'frame_shift': 10
}
SPECTRUM_DEFAULT_SETTING = {
    'window': torch.hann_window(400), 'normalized': False, 'pad': 0 # the parameter in window is win_length
}

# random factory
RANDOM_FACTOR = {
    'beta':np.random.beta, 'uniform': np.random.uniform, 'int': np.random.randint, 'random': np.random.random
}

# transform np arrary as python list
TRANSFORM_FACTOR ={
    'nptolist': lambda x: x.tolist() if isinstance(x, np.ndarray) else x
}

# DITHER_RANGE used to trim wav 10 means the speech feats are in frame(1s=>100 frame) level; 
# 1600 means feats are waveforme level(sample point 1s=>16000 samples)
DITHER_RANGE = [10, 1600]

# Recompile pattern
RE_PATTERN = {'space': re.compile(r" +"), 'dot': re.compile(r"\.")}

# read data list
def read_list(list_file):
    d_list = []
    with open(list_file, encoding='utf-8') as lf:
        for line in lf.readlines():
            d_list.append(line.strip())
    lf.close()
    return d_list


# split str "0.1 0.2 0.3" to float [0.1 0.2 0.3] 
# str to int; int to sym; tensor to str
def sym2float(sym_list):
    int_list = list(map(lambda x: float(x), sym_list.split(" ")))
    return int_list
def int2sym(int_list):
    if not isinstance(int_list, list):
        int_list = [int_list]
    int_list = [str(x) for x in int_list]
    return int_list
def sym2int(sym_list):
    int_list = list(map(lambda x: int(x), sym_list.split(" ")))
    return int_list
def tensor2str(t):
    if isinstance(t, torch.Tensor):
        t = t.numpy()
    t = list(t)
    t = list(map(lambda x: str(x), t))
    return t

# save wav as PCM_S 16bit 16k: always use to test code
def save_wav(wav, names):
    if isinstance(names, list):
        names = "_".join(names)
    torchaudio.save(
        "{}.wav".format(names), wav, sample_rate=16000, encoding="PCM_S", bits_per_sample=16
    )

# Splice feats: append context 
def splice_feats(feats, left_context, right_context, seq=False):
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
def max_energy(wav, frame_length=400, hop_length=160, shift_type='mid'):
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
def unfold_list(lst):
    if not isinstance(lst, list):
        lst = [lst]
    l = _unfold_list(lst)
    trans = int if re.search(RE_PATTERN['dot'], l) == None else float
    l = [trans(i) for i in l.split(" ") if i !=""]
    return l
# sub method of unfold_list 
def _unfold_list(lst):
    new = ""
    for x in lst:
        if isinstance(x, list):
            x = _unfold_list(x)
        x = str(x)
        new = new + x + " "
    return new

# random one sample from a pool
def random_one(pools, pool_len):
    return (pools[random.randint(0, pool_len-1)])


# spec augmentation
def spec_augment(spec, num_t_mask=2, num_f_mask=2, max_t=20, max_f=10):
    assert isinstance(spec, torch.Tensor)
    aug_spec = spec.clone().detach()
    max_frames = aug_spec.size(0)
    max_freq = aug_spec.size(1)
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
    return aug_spec

# Speech augmentation: reverb, add noise, change speed
def wav_augment(waveform, config, rirs=None):
    # add noise reverb change speech NOTE: noise here means none humanc speech voice!!!!
    # TODO: if change speech alignment and segment should change too!!!
    if config.get("volume", False):
        volume_sampler = config['volume']['sampler']
        volume_sampler_config = config['volume']['config']
        ratio = RANDOM_FACTOR[volume_sampler](**volume_sampler_config)
        waveform = ratio * waveform

    if config.get("white_niose", False):
        noise_config = config['white_noise']['config']
        gua = torch.normal(**noise_config, size=waveform.size())
        waveform = waveform + gua
    
    if rirs:
        rirs_prob = config.get('rirs_prob', 0.4)
        if random.uniform(0,1) < rirs_prob:
            _, rirs_src = wavfile.read(rirs)
            rirs_src = rirs_src / np.sqrt(np.sum(rirs_src**2))
            rirs_src = rirs_src.astype(np.float32)
            l = waveform.size(1)
            waveform = waveform[0].numpy()
            waveform = signal.convolve(waveform, rirs_src)[:l]
            waveform = torch.from_numpy(waveform)
            waveform = waveform.view(1,-1)
    return waveform


# make mix wav
def make_mix_wav(wavs, ratios, mean_dur='max', major=False, skip_idx=None, delay_prob=0.35):
    if skip_idx != None:
        skip_feats = wavs[-skip_idx:]
        skip_ratios = ratios[-skip_idx:]
        wavs = wavs[:-skip_idx]
        ratios = ratios[:-skip_idx]
    else:
        skip_feats, skip_ratios = [],[]

    # compute rms before padding wave
    rms = [w.norm(p=2).item() for w in wavs] # compute energy
    rms = list(map(lambda x: x if x > 0.01 else 1, rms)) # avoid devide very small value 0
    max_rms = max(rms) # find the max energy
    
    # make wav delay
    #if random.randint(0, 10) % 3 == 0:
    if (random.random() < delay_prob) and (len(wavs)>1):
        for delay_idx in range(1, len(wavs)): # apply delay from the second wavform
            if delay_idx == 1:
                delay_len = random.randint(16000, 16000*3)
            else:
                delay_len = delay_len + random.randint(16000, 16000*3)
            zero_padding = torch.zeros(1, delay_len)
            wavs[delay_idx] = torch.cat([zero_padding, wavs[delay_idx]], dim=1)

    # compute wav size
    wav_size = [wav.size(1) for wav in wavs]
    if mean_dur == 'max':
        wav_len = max(wav_size) if max(wav_size) >= 16000 else 16000
    else:
        wav_len = wav_size[0]
    
    # padding wav
    wavs = [
        padding_wav(wavs[i], length=wav_len) for i in range(len(wavs))
    ]

    # control the target wav always lounder one or weak one
    if major:
        assert len(wavs) > 1
        if major == 1:
            ratios[0], ratios[1] = (ratios[1], ratios[0]) if ratios[0]<ratios[1] else (ratios[0], ratios[1])
        else:
            ratios[1], ratios[0] = (ratios[1], ratios[0]) if ratios[0]<ratios[1] else (ratios[0], ratios[1])
    
    # scale wavs 
    wavs = [wavs[i]*(max_rms/rms[i]) for i in range(len(rms))] # make wav1 wav2 to equal energy
    scaled_wav = [wavs[i]*ratios[i] for i in range(len(wavs))]
    mix_wav = sum(scaled_wav)

    if torch.max(mix_wav) > 1 or torch.min(mix_wav) < -1:
        ratios = [r/sum(ratios) for r in ratios]
        scaled_wav = [wavs[i]*ratios[i] for i in range(len(wavs))]
        mix_wav = sum(scaled_wav)
    return [mix_wav] + scaled_wav + skip_feats, ratios + skip_ratios


# padding wav with zeros
def padding_wav(wav, length, length_idx=1):
    if wav.size(length_idx) < length:
        wav_shape = wav.size()
        res_num = length - wav.size(length_idx) 
        if length_idx == 0:
            d = wav_shape[1]
            padding_shape = (res_num, d)
        else:
            d = wav_shape[0]
            padding_shape = (d, res_num)
        padding = torch.zeros(padding_shape)
        wav = torch.cat([wav, padding], dim=length_idx)
    else:
        wav = wav[:,:length] if length_idx == 1 else wav[:length,:]
    return wav


# trim wav by lenth and segment
def trim_wav(wav, segment):
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

# convert segment info as alignment
def convert_mfa_to_align(mfa_obj):
    ali = []
    # mfa_obj zip([p1, p2], [p1_head, p1_tail, p2_head, p2_tail]) => (phn_list, positio
    # for idx, p in [p1, p2] i is index in phone list it also can be used to
    # find the position of the corresponding phone as index*2, index*2+1 which is p1_head, p1_tail
    for (phn, pos) in mfa_obj:
        one_ali = [
            p for i, p in enumerate(phn)
            for _ in range(pos[i * 2], pos[i * 2 + 1] + 1)
        ]
        ali.extend(one_ali)
    return torch.tensor(ali)


# make corruption pairs, maybe triple or more
# detach_f: detach_functions, for self corruption the datalist will format as json object so detach_f will be json.loads
# for none target corruption detach_f is lambda x: x
def make_corrupt_party(
        corrupt_list, corrupt_list_len, config, corruption_material, detach_f=lambda x: x, prob=1.2, 
    ):
    ratios = []
    n = config.get('num_corrupt')
    if config.get('random_num', False):
        assert(n > 1)
        n = RANDOM_FACTOR['int'](0, n+1)
    #if config.get('prob', 1.2) < 1:
    #    dice = random.uniform(0,1)
    #    if dice > config.get('prob'):
    #        n = 0

    sampler = config.get('sampler')
    assert (sampler in RANDOM_FACTOR.keys())
    sampler_config = config.get('sampler_config')
    if len(corruption_material) == 0:
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)
        c_idx = 0
    else:
        c_idx = len(corruption_material)
    
    for i in (range(n)):
        one_corrupt = corrupt_list[np.random.randint(0, corrupt_list_len)]
        one_corrupt = detach_f(one_corrupt)
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)
        if isinstance(one_corrupt, dict):
            corruption_material[c_idx+i+1] = one_corrupt
        elif isinstance(one_corrupt, str):
            corruption_material[c_idx+i+1] = {'sph': one_corrupt}
        else:
            raise NotImplementedError("corrupt list error")
    return ratios, corruption_material, n


# time shiftting: shiftting keywords from waveforme to make sure that: when mix tow keywords they will completely overlaped
def time_shifting(wav, frame_length=400, hop_length=160, shift_type='mid'):
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
def make_segment(segments, win_len, trim_type='raw', dither=False, sample_rate=16000):
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
            raise NotImplementedError("Only support raw, xxx")
    return seg_head_tail, idx_head_tail


# detach corruption
def detach_corruption(material, segment_idx=None):
    keywords = [] 
    phn_labels = []
    bpe_labels = []
    labels = []
    for i, (_idx, info) in enumerate(material.items()):
        if 'word_keyword' in info:
            keywords.append(info['word_keyword'])
        if 'keyword' in info:
            keywords.append(info['keyword'])
        if 'label' in info:
            label = info['label']
            if segment_idx != None:
                segment_head, segment_tail = segment_idx[i]
                label = label[segment_head: segment_tail] 
            labels.append(label)
        if 'phn_label' in info:
            phn_label = info['phn_label']
            if segment_idx != None:
                segment_head, segment_tail = segment_idx[i]
                phn_label = phn_label[segment_head: segment_tail] 
            phn_labels.append(phn_label)
        if 'bpe_label' in info:
            bpe_label = info['bpe_label']
            if segment_idx != None:
                segment_head, segment_tail = segment_idx[i]
                bpe_label = bpe_label[segment_head: segment_tail] 
            bpe_labels.append(bpe_label)
    return keywords, labels, phn_labels, bpe_labels

# insert special token in label sequence such as SOS: 0(start of sentence) 
# 1 2 3 4 5 -> "0" 1 2 3 4 5
def inject_special_token(
        keyword, keyword_length, label=None, 
        positive=True, keyword_pos=None, special_token={}, bpe_label=None, bpe_candidate=None
    ):
    TEXT_SPEC_TOKEN.update(special_token)
    new_phn_label = copy.deepcopy(label)
    new_bpe_label = copy.deepcopy(bpe_label)
    new_keyword = copy.deepcopy(keyword)
    if (not positive) and (TEXT_SPEC_TOKEN['punk'] != None):
        new_phn_label = torch.tensor([TEXT_SPEC_TOKEN['punk'] for x in range(len(new_phn_label)//3)])
        new_bpe_label = torch.tensor([TEXT_SPEC_TOKEN['unk']  for x in range(len(new_bpe_label)//3)])

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
        bpe_kw_head = bpe_candidate[keyword_pos]
        bpe_kw_tail = bpe_candidate[keyword_pos+keyword_length]
        bpe_kw = bpe_label[bpe_kw_head: bpe_kw_tail]
        bpe_kw.insert(0, [TEXT_SPEC_TOKEN['sok']])
        bpe_kw.insert(len(bpe_kw), [TEXT_SPEC_TOKEN['eok']])
        new_bpe_label[bpe_kw_head: bpe_kw_tail] = bpe_kw

    return new_keyword, new_phn_label, new_bpe_label, keyword_pos

# snipe_edges for waveform
def snipe_edge(waveform, hop_length=160):
    num_samples = waveform.size(1)
    edges = num_samples % hop_length
    return waveform[:,0:num_samples-edges]

# process raw json line
# data list is aranged in json format, in this function convert json into dict
# NOTE:  egs_format is a test feature i.e. read data from egs file just same like kaldi
# But egs_format didn't boost the training speed yet. just keep it and waiting for tuning
def process_raw(data):
    for sample in data:
        one_sample = json.loads(sample['src'])
        if 'self_corruption' in sample:
            self_corruption = sample['self_corruption']
            self_corruption = [json.loads(d) for d in self_corruption]
            one_sample.update({'self_corruption': self_corruption})
        if 'none_target_corruption' in sample:
            none_target_corruption = sample['none_target_corruption']
            one_sample.update({'none_target_corruption': none_target_corruption})
        if 'rirs' in sample:
            rirs_src = sample['rirs']
            one_sample.update({'rirs':rirs_src})
        if 'neg_candidate' in sample:
            neg_candidate = sample['neg_candidate']
            one_sample.update({'neg_candidate': neg_candidate})
        epoch = sample['epoch']
        one_sample.update({'epoch': epoch})
        yield one_sample

# make corruption
# mix wav: 
#    - self corrution means mix two target speech: such as keyword1 speech + keyword2 speech
#    - none_target corruption means target speech with noise: such as keyword1 speech + none target inteferance
# NOTE: in this function, waveform are not mixed !!!  Just extract mix mmaterials !!! e.g.:
# corruption_material:{1: keyword1 speech FILE, 2: keyword2 speech FILE, 3: niose speech FILE}
# corruption ratios: [0.1, 0.6, 0.5]    
# the function is process_speech_feats:mix_wav will employ corruption material and corruption ratios to make
# the real mix waveform!!!!
def process_corruption(data, config, egs_format=False):
    for sample in data:
        corruption_material = {}
        corruption_ratios = []
        num_corrupt = n_scorrupt = n_ncorrupt = 0
        if config.get('self_corruption', False): # make self corruption materials
            assert 'self_corruption' in sample
            corrupt_list = sample['self_corruption']
            corrupt_list_len = len(corrupt_list) 
            ratios, corruption_material, n_scorrupt = make_corrupt_party(
                corrupt_list, corrupt_list_len, config['self_corruption'], corruption_material, 
            )
            corruption_ratios.extend(ratios)
            num_corrupt += n_scorrupt
        
        if config.get('none_target_corruption', False): # make none target corruption materials
            assert 'none_target_corruption' in sample
            corrupt_list = sample['none_target_corruption']
            corrupt_list_len = len(corrupt_list)
            ratios, corruption_material, n_ncorrupt = make_corrupt_party(
                corrupt_list, corrupt_list_len, config['none_target_corruption'], corruption_material
            )
            corruption_ratios.extend(ratios)
            num_corrupt += n_ncorrupt

        # save the metarial into sample dict
        sample.update({
            'corruption_ratios': corruption_ratios,
            'corruption_material': corruption_material,
            'n_scorrupt': n_scorrupt,
            'n_ncorrupt': n_ncorrupt,
            'num_corrupt': num_corrupt,
        })
        yield sample

# process speech feats
# load wav -> corrupt wav -> destroy a positive sample to negative (made for FA) -> extract fbank
# NOTE: this function can support load kaldi ark feats, torch pt file and read wavefrom from raw wav file
# NOTE: But kaldi feat, torch pt have not been verified in training process be carefull that.
def process_speech_feats(data, config, egs_format=False):
    for sample in data:
        input_data_type = config.get('data_type', 'raw') # feats type: raw=>waveform kaidl: kaldi ark, pt: torch.pt
        if (input_data_type != 'raw') and ('corruption_material' in sample): # corruption only support performed on waveform
            raise NotImplementedError("Only support corruption on waveforme")
        feats = [sample['sph']]

        if 'corruption_material' in sample: # corruption: self corruption=>mix training none target corruption=> data augmentation
            corruption_material = sample['corruption_material']
            corruption_feats = [corruption_material[x]['sph'] for x in corruption_material.keys()]
            corruption_segments = [
                corruption_material[x]['segment'] if 'segment' in corruption_material[x] else -1
                for x in corruption_material.keys()
            ]
            feats.extend(corruption_feats)
        
        if egs_format:
            input_data_type = 'empty'
        feats = [INPUT_DATA_LOADER[input_data_type](x) for x in feats]
        feats = [INPUT_DATA_LOADER['rm_sr'](x) for x in feats] if input_data_type == 'raw' else feats

        # Wav augment: volume change and add white noise
        if config.get('wav_augment', False):
            if 'rirs' in sample:
                rirs_src = sample['rirs']
            else:
                rirs_src = None
            feats = [wav_augment(f, config.get('wav_augment'), rirs_src) for f in feats]

        # Trim wav according to segment 
        if config.get('trim_config', False):
            if 'segment' in sample:
                segments = sample['segment']
            else:
                segments = [[0,0]]
            if 'corruption_segments' in locals().keys():
                segments += corruption_segments
            segments, segment_idx = make_segment(segments, **config.get("trim_config", {}))
            sample.update({"segment_idx": segment_idx})
            feats = [trim_wav(feats[i], segments[i]) for i in range(len(feats))]
        
        # Mix wav feats
        if 'corruption_material' in sample:
            mix_config = config.get('mix_config', {})
            if (sample['n_scorrupt'] != 0) and (sample['n_ncorrupt'] != 0): 
                # target1 speech + target2 speech + corruption speech
                # random select a target and mix it with noise: target1 + corruption speech || target2 + corruption speech
                n_ncorrupt = sample['n_ncorrupt']
                crpt_idx = random.randint(0, sample['n_scorrupt'])
                target_feats = feats[crpt_idx]
                noise_feats, noise_ratio = feats[-n_ncorrupt:], sample['corruption_ratios'][-n_ncorrupt:]
                tmp_ratio = [sample['corruption_ratios'][crpt_idx]] + noise_ratio
                mix_feats, _ = make_mix_wav([target_feats]+noise_feats, tmp_ratio, **mix_config)
                feats[crpt_idx] = mix_feats[0] # replace the selected target speech by mixed noisy speech
                #feats, sample['corruption_ratios'] = feats[:-n_ncorrupt], sample['corruption_ratios'][:-n_ncorrupt]
                skip_idx = n_ncorrupt # here noise has been mix into one of the target speech so in the following mix
                                      # mixing process skip noise speech !!!!NOTE!!!!
            else:
                skip_idx = None

            feats, sample['corruption_ratios'] = make_mix_wav(
                feats, sample['corruption_ratios'], skip_idx=skip_idx, **mix_config
            )
            #sample.update({"wav": copy.deepcopy(feats)})
        if config.get('snipe_edge', False):
            hop_length = config['snipe_edge'].get('hop_length', 160)
            feats = [snipe_edge(f, hop_length) for f in feats]
        
        if config.get('return_raw', False):
            if 'corruption_material' in sample:
                raw_feats = copy.deepcopy(feats[1])
            else:
                raw_feats = copy.deepcopy(feats[0])
            sample.update({'raw_wav': raw_feats.squeeze(0)}) # only keep the target speech
        
        if config.get('random_raw', False):
            sample_len1 = feats[1].size(1)
            dur = 16000 * 4 
            sample_head = random.randint(0, sample_len1-dur-1) if sample_len1 > dur else 0
            raw_wav1 = copy.deepcopy(feats[1][:, sample_head:sample_head+dur])
            sample.update({'raw_wav1': raw_wav1.squeeze(0)})  

            sample_len2 = feats[2].size(1)
            dur = 16000 * 4 
            sample_head = random.randint(0, sample_len2-dur-1) if sample_len2 > dur else 0
            raw_wav2 = copy.deepcopy(feats[2][:, sample_head:sample_head+dur])
            sample.update({'raw_wav2': raw_wav2.squeeze(0)})  
        # Extract feature: MFCC / FBANK 
        feats_type = config.get('feats_type', 'fbank')
        feats_config = config.get('feats_config', FBANK_DEFAULT_SETTING)
        feats = [FEATS_EXTRACTOR[feats_type](f, **feats_config) for f in feats]


        # Splice Feature: add context
        if config.get('splice_config'):
            splice_config = config.get('splice_config')
            feats = [splice_feats(f, **splice_config) for f in feats]

        # Subsample Feature: skip frame
        if config.get('subsample_rate'):
            feats = [f[::config.get('subsample_rate')] for f in feats]

        # Load feats into torch Tensor
        start_idx = 0
        if 'corruption_material' in sample:
            mix_feats = feats[start_idx]
            start_idx += 1
            sample.update({"mixspeech": mix_feats})
        else: # if no corruption meterail the 1th feats is clean feats
            sample.update({"speech": feats[0]})
        
        # keep clean feats e.g. mix_wav = wav1 + wav2 the following code will 
        # concat wav1, wav2 into one matrix and load to {speech: [wav1; wav2]}
        if sample.get("n_scorrupt", 0) > 0:
            clean_feats = feats[start_idx: start_idx+sample['n_scorrupt']+1]
            ratios = sample['corruption_ratios']
            #TODO: consider about add noise augment ratios
            ratios = ratios[0:sample['n_scorrupt']+1]
            clean_feats = torch.cat([x.unsqueeze(0) for x in clean_feats], dim=0)
            start_idx = sample['n_scorrupt'] + 1
            sample.update({"speech": clean_feats[0]}) # here only keep the first wav
            sample.update({"ratios": ratios})
        # Same with the code upper noise wav will load to {noise: niose_wav} 
        # NOTE: noise wav have not involved in training process, in practice noise wav only 
        # applied in former HOMO method, we comment this code to save the memory resource
        # if sample.get("n_ncorrupt", 0) > 0:
        #     noise_feats = feats[start_idx: start_idx+sample['n_ncorrupt']]
        #     if len(noise_feats) > 1:
        #         noise_feats = torch.cat([x.unsqueeze(0) for x in noise_feats], dim=0)
        #     else:
        #         noise_feats = noise_feats[0]
        #     sample.update({"niose_speech": noise_feats})
        yield sample 

# Process text feats, mainly deal with segment:
# e.g. in process_speech_feats wav has been trimed by segment, and the text label will 
# be cutted in this function acorrding to segment also.
def process_text_feats(data, neg_token=None, sc_token=None):
    for sample in data:
        if 'bpe_label' in sample:
            feak_pad_label = sample['bpe_label'] if not neg_token else neg_token
        else:
            feak_pad_label = [0]
        if ('label' in sample) and ('segment_idx' in sample):
            label = sample['label']
            segment_idx = sample['segment_idx']
            m_head, m_tail = segment_idx[0]
            label = label[m_head: m_tail]
            sample.update({'label': label})
        
        if 'corruption_material' in sample:
            if sc_token:
                fifo_label = sample['bpe_label']
            if 'segment_idx' in sample:
                c_segment_idx = sample['segment_idx'][1:]
            else:
                c_segment_idx = None
            c_keyword, c_label, c_phn_label, c_bpe_label = detach_corruption(sample['corruption_material'], c_segment_idx)

            if len(c_keyword) != 0:
                sample.update({"mix_keyword": sample['word_keyword']+unfold_list(c_keyword)})
            if len(c_label) != 0:
                sample.update({"crpt_label": c_label})

            if len(c_phn_label) != 0:
                for x in range(len(c_phn_label)):
                    sample.update({"c_phn_label{}".format(x): c_phn_label})
            
            if len(c_bpe_label) != 0:
                n_label = 1 + len(c_bpe_label)
                for x in range(len(c_bpe_label)):
                    sample.update({"c_bpe_label{}".format(x): c_bpe_label[x]})
                    feak_pad_label = c_bpe_label[x] if not neg_token else neg_token
                    if sc_token:
                        fifo_label = fifo_label + [sc_token] + c_bpe_label[x]
            else:
                n_label = 1
            
            #TODO: assume max mix is 3
            label_mask = [0 for x in range(n_label)]
            n_pad_label = 2 - len(c_bpe_label)
            for x in range(n_pad_label):
                sample.update({"c_bpe_label{}".format(x+len(c_bpe_label)): feak_pad_label})
                sample.update({"c_phn_label{}".format(x+len(c_phn_label)): feak_pad_label})
                label_mask  = label_mask + [1]
            if sc_token:
                sample.update({'fifo_label': copy.deepcopy(fifo_label)})
            sample.update({'n_label': n_label})
            sample.update({'label_mask': label_mask})
        yield sample


# Process: sample keyword from continues label
def process_sampled_keyword_from_label(
        data, positive_prob=0.5, neg_len=None, special_token={}, id2phone=None, lexicon=None
):
    # TEXT_SPEC_TOKEN = {'sos','eos','sok', 'eok', 'unk'}
    # sos: start of setence, eos: end of setence, sok: start of keyword, eok, end of keyword, unk: unknow token
    TEXT_SPEC_TOKEN.update(special_token)
    for sample in data:
        new_phn_label = copy.deepcopy(sample['phn_label'])
        new_bpe_label = copy.deepcopy(sample['bpe_label'])
        bpe_candidate = copy.deepcopy(sample['b_kw_candidate'])
        kw, kw_pos, kw_length, pos, target = make_keyword(sample, positive_prob, neg_len=neg_len)
        kw, new_phn_label, new_bpe_label, kw_pos = inject_special_token(
            keyword=kw, keyword_length=kw_length, positive=pos, label=new_phn_label, 
            keyword_pos=kw_pos, special_token=special_token,  bpe_label=new_bpe_label, bpe_candidate=bpe_candidate
        )

        sample.update({'keyword': kw, 'phn_label': new_phn_label, 'bpe_label': new_bpe_label, 'target': target}) 
        yield sample

# process permuate label
def process_permuate_label(data, sc_token=None):
    for sample in data:
        labels = []
        labels.append(copy.deepcopy(sample['bpe_label']))
        for x in range(1, sample['n_label']):
            labels.append(sample['c_bpe_label{}'.format(x-1)])
        permuates = permuate_labels(labels, sc_token)
        for idx, per in enumerate(permuates):
            sample.update({"per{}".format(idx): per})
        yield sample


# process sot label, i.e. add sc token into label
def process_sot_label(data, special_token=None):
    #for i, sample in enumerate(data):
    for sample in data:
        new_label = copy.deepcopy(sample['bpe_label'])
        if 'sc' not in special_token:
            print('sc should be specify for sot methods')
        crpt_bpe_label = sample['c_bpe_label']
        new_label = new_label + [special_token['sc']] + crpt_bpe_label
        sample.update({'bpe_label': new_label})
        yield sample

# process cohort label
def process_cohort_label(data):
    for sample in data:
        bpe_label = copy.deepcopy(sample['bpe_label'])
        phn_label = copy.deepcopy(sample['phn_label'])
        if 'c_bpe_label' not in sample:
            sample.update({"c_bpe_label": bpe_label})
        if 'c_phn_label' not in sample:
            sample.update({"c_phn_label": phn_label})
        yield sample

# process sequential rec
def process_sequentail_label(data, special_token=None):
    TEXT_SPEC_TOKEN.update(special_token)
    for sample in data:
        keywords = []
        # first sample keywords from bpe_label
        dice = random.uniform(0, 1)
        label_mask = []
        max_label_len = 0
        if dice >= 0.5:
            if len(sample['bpe_label']) > max_label_len:
                max_label_len = len(sample['bpe_label'])
            one_kw, _, kw_length, _, _ = make_keyword(sample, positive_prob=1.2, neg_len=None)
            one_kw, _, _, _ = inject_special_token(one_kw, kw_length, special_token=special_token)
            keywords.append(one_kw)
            label_mask.append(1)
        
        for idx in range(sample['n_label']-1):
            dice = random.uniform(0,1)
            if dice < 0.5:
                label_mask.append(0)
                continue
            s_crpt_idx = idx + 1
            one_crpt = sample['corruption_material'][s_crpt_idx]
            if len(one_crpt['bpe_label']) > max_label_len:
                max_label_len = len(one_crpt['bpe_label'])
            one_kw, _, kw_length, _, _ = make_keyword(one_crpt, positive_prob=1.2, neg_len=None)
            one_kw, _, _, _ = inject_special_token(one_kw, kw_length, special_token=special_token)
            keywords.append(one_kw)
            label_mask.append(1)
        if len(keywords) == 0:
            keywords = [special_token['unk']]

        if len(label_mask) < 3:
            label_mask = label_mask + [1 for x in range(3-len(label_mask))]
        
        if sum(label_mask) == 3:
            label_mask.append(0)
        else:
            label_mask.append(1)
        one_neg_label = [special_token['unk'] for x in range(max_label_len)]
        sample.update({'keyword': keywords, 'label_mask': label_mask, 'neg_label': one_neg_label}) 
        yield sample

# sample keyword from asr label, actually sample positive and make a negative
def make_keyword(sample, positive_prob, neg_len=None):
    dice = random.uniform(0, 1)
    label = copy.deepcopy(sample['phn_label'])
    if dice > positive_prob: # negative sample
        if 'c_phn_label' in sample:
            crpt_label = sample['c_phn_label']
            mlabel = label + crpt_label
        kw = random_one_neg(sample['neg_candidate'], neg_len, mlabel, sample['key'].split('-')[0])
        kw_pos = -1
        pos = False
        target = torch.tensor([0])
    else: # positvie sample
        kw_candidate = sample.get('kw_candidate', None)
        kw, kw_pos = sample_kw_from_label(label, kw_candidate)
        pos = True
        target = torch.tensor([1])
    return kw, kw_pos, len(kw), pos, target

# sample positive keyword from asr label
def sample_kw_from_label(label, kw_candidate=None):
    kw_len = random.randint(2, 6)
    if kw_candidate: #TODO: a little bit confuse ...  optim it latter
        kw_len = kw_len if kw_len < len(kw_candidate) else 1
        kw_pos_idx = random.randint(0, len(kw_candidate)-kw_len-1) if len(kw_candidate) > kw_len+1 else 0
        kw_pos = kw_candidate[kw_pos_idx]
        if kw_pos_idx+kw_len >= len(kw_candidate):
            kw_len -= 1 
        kw_len = kw_candidate[kw_pos_idx+kw_len] - kw_pos
    else:
        kw_pos = random.randint(0, len(label)-kw_len) if len(label) > kw_len else 0
    kw = label[kw_pos: kw_pos + kw_len]
    return kw, kw_pos

# sample negative keyword from the whole corpus
def random_one_neg(neg_list, neg_len, pos_label, spk_id=None):
    neg = pos_label[0]
    flatten_label = unfold_list(pos_label)
    flatten_neg = unfold_list(neg)
    flatten_label = int2sym(flatten_label)
    flatten_neg = int2sym(flatten_neg)
    if spk_id != None:
        neg_spk = spk_id
    else:
        neg_spk = -1
    while (" ".join(flatten_neg) in " ".join(flatten_label)) or (neg_spk == spk_id):
        one_neg_list = neg_list[random.randint(0, neg_len-1)]
        one_neg_list = json.loads(one_neg_list)
        if spk_id != None:
            neg_spk = one_neg_list['key'].split('-')[0]
        one_neg_label = one_neg_list['phn_label']
        kw_candidate = one_neg_list.get('kw_candidate', None) 
        neg, _ = sample_kw_from_label(one_neg_label, kw_candidate)
        flatten_neg = unfold_list(neg)
        flatten_neg = int2sym(flatten_neg)
    return neg


# process fix keyword from segment
def process_fix_keyword(data, special_token={}):
    for sample in data:
        if len(special_token) != 0:
            kw = sample['keyword']
            label = sample['label']
            kw, label, _ = inject_special_token(
                keyword=kw, keyword_length=len(kw), label=label, special_token=special_token
            )
            sample.update({'keyword': kw, 'label': label})
        yield sample


# all the label information are [[...], [...]] unfold them.
# NOTE: the label in raw json file is aranged in word format especially for chinese characters such as :
# [[1,2],[3],[4,5]] is this example [1,2] is a chinese word such as [你好], this kind of arrangement is usefull
# for sample keyword as when sample index 0 [1,2] will be the candidate keyword but not [1]
# [process_list_data] is aim to unfold the label sequence, use the example above again. [[1,2],[3],[4,5]] ->
# [1,2,3,4,5], this function is applied after sample keywords
def process_list_data(dataset):
    for sample in dataset:
        for key, value in sample.items():
            if key in NONE_TENSOR_KEY:
                continue
            if isinstance(value, list):
                value = unfold_list(value)
            if not isinstance(value, torch.Tensor):
                sample.update({key: torch.tensor(value)})
        yield sample


# such as ctc loss and rnn-t loss need speech length and target length
# but after make batch. these data will be append as the same length.
# so we compute length information before make batch
def make_length(dataset):
    for sample in dataset:
        length_info = {}
        for key, value in sample.items():
            if key in NONE_TENSOR_KEY:
                continue
            if not isinstance(value, torch.Tensor):
                continue
            if value.dim() == 0:
                continue
            new_key = "{}_len".format(key)
            length = value.size(0)
            length_info.update({new_key: torch.tensor(length)})
        sample.update(length_info)
        yield sample


# concat into one batch
def concat_tensor(data_list, seq_padding=False, padding_value=0):
    if seq_padding:
        tensor = pad_sequence(data_list, batch_first=True, padding_value=padding_value)
    else:
        tensor = torch.cat([x.unsqueeze(0) for x in data_list], dim=0)
    return tensor


# fetch keys
# there are a lot of inter material is data processing, however most of them are not 
# training ingredients so we fetch the training data by keys, more detail can be found in
# config files fetch_keys: 
def fetch_tensor(data, fetch_key):
    for sample in data:
        if fetch_key[0] == 'key':
            sort_key = fetch_key[1]
        else:
            sort_key = fetch_key[0]
        index = torch.tensor([x[sort_key].size(0) for x in sample])
        index = torch.argsort(index, descending=True)
        return_feats = []
        for k in fetch_key: #TODO: this code in not safe ...
            if (k == 'key') or (k == 'tag'):
                return_feats.append([sample[i][k] for i in index])
                continue
            if k in NONE_TENSOR_KEY:
                continue
            if sample[0][k].dim() != 0:
                seq_padding = True
            else:
                seq_padding = False
            if k in CTC_KEY:
                padding_value = -1
            else:
                padding_value = 0
            return_feats.append(
                concat_tensor(
                    [sample[i][k] for i in index], seq_padding=seq_padding, padding_value=padding_value
                )
            )
        yield tuple(return_feats)


# make batch
# TODO: make it support batch bce
def make_batch(data, batch_size=256):
    buf = []
    batch_count = {}
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf
