# ref: wenet processor.py
import re
import random
import torch
import torchaudio
import json
import copy

import data.loader.kaldi_io as kaldi_io
import torchaudio.compliance.kaldi as kaldi
import numpy as np
import numpy as np
import soundfile as sf

from torch.nn.utils.rnn import pad_sequence
#from warprnnt_pytorch import rnnt_loss

# input loader and feats extractor mapping
INPUT_DATA_LOADER = {
    'raw': torchaudio.load, 'kaldi': kaldi_io.read_mat, 'torch': torch.load,
    'rm_sr': lambda x: x[0]
}
FEATS_EXTRACTOR = {
    'mfcc': kaldi.mfcc, 'fbank': kaldi.fbank, 
    'spectrum': torchaudio.functional.spectrogram, 'empty': lambda x: x 
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

# random factor
RANDOM_FACTOR = {
    'beta':np.random.beta, 'uniform': np.random.uniform, 'int': np.random.randint, 'random': np.random.random
}

TRANSFORM_FACTOR ={
    'nptolist': lambda x: x.tolist() if isinstance(x, np.ndarray) else x
}
# DITHER_RANGE used to trim wav 10 means the speech feats are in frame(1s=>100 frame) level; 1600 means feats are waveforme level(sample point 1s=>16000 samples)
DITHER_RANGE = [10, 1600]

# read data list
def read_list(list_file):
    d_list = []
    with open(list_file, encoding='utf-8') as lf:
        for line in lf.readlines():
            d_list.append(line.strip())
    lf.close()
    return d_list


# str to int
def sym2int(sym_list):
    int_list = list(map(lambda x: int(x), sym_list.split(" ")))
    return int_list

# split str "0.1 0.2 0.3" to float [0.1 0.2 0.3]
def sym2float(sym_list):
    int_list = list(map(lambda x: float(x), sym_list.split(" ")))
    return int_list

# save wav as PCM_S 16bit 16k: always use to test code
def save_wav(wav, names):
    torchaudio.save(
        "{}_{}.wav".format(*names), wav, sample_rate=16000, encoding="PCM_S", bits_per_sample=16
    )
# detach data from json obj
def detach_json(obj):
    r_dict = {}
    if 'sph' in obj:
        r_dict.update({"sph": obj['sph']})
    if 'candidate_kw_idx' in obj:
        idx = obj['candidate_kw_idx'][0]
        segment_head = obj['segment'][idx][0]
        r_dict.update({"segment": segment_head})
    if 'word_keyword' in obj:
        r_dict.update({"word_keyword": obj['word_keyword']})
    return r_dict
# read json line and transfer it into a dict
def make_raw_sample_dict(json_obj):
    keys = [
        'key', 'sph', 'word_label', 'phone_label', 'align', 'word_keyword',
        'phone_keyword', 'word_neg_keyword', 'phone_neg_keyword', 'segment',
        'keyword_idx', 'candidate_kw_idx', 'word_label_aux', 'phone_label_aux',
        'tag', 'plot_target_id', 'inter', 'speaker'
    ]
    #TODO: add common segment for double speech input
    one_sample = {}
    for k in keys:
        if k in json_obj.keys():
            one_sample.update({k: json_obj[k]})
        if 'inter' in one_sample:
            inter_pool = one_sample['inter']
            idx = random.randint(0, len(inter_pool)-1)
            one_inter = inter_pool[idx]
            target_utt = one_sample['sph']
            one_sample.update({
                'sph': "{};{}".format(target_utt, one_inter)
            })

    return one_sample


# flatten list [[1,2,3],[4,5,6]] => [1,2,3,4,5,6]
def unfold_list(lst):
    new = []
    for x in lst:
        if isinstance(x, list):
            new.extend(x)
        else:
            new.append(x)
    return new


# add special token into transcription label
def inject_special_token(
    org_seq,
    head_token=None,
    tail_token=None,
):
    assert (isinstance(org_seq, list))
    ht = [head_token] if head_token != None else []
    tt = [tail_token] if tail_token != None else []

    if not isinstance(org_seq[0], list):
        # for keyword special token keyword = [1,2,3]
        # after process keyword = [ht,1,2,3,tt]
        return ht + org_seq + tt
    else:
        # for transcription special token label=[[1,2,3],[4,5,6]]
        # after process label = [[ht,1,2,3,tt],[ht,4,5,6,tt]]
        new_seq = []

        for w in org_seq:
            w = ht + w + tt
            new_seq.append(w)
        return new_seq

# random one sample from a pool
def random_one(pools, pool_len):
    return (pools[random.randint(0, pool_len-1)])


# spec augmentation
def spec_augment(spec, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10):
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
    yield aug_spec


# speech augmentation: reverb, add noise, change speed
def wav_augment(waveform, config, k=None):
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
    if k:
        torchaudio.save("{}_{}.wav".format(k, ratio), waveform, sample_rate=16000, encoding="PCM_S", bits_per_sample=16)
    return waveform

# make mix wav
def make_mix_wav(wavs, ratios, segment_info=None, win_len=1, trim_dither=False):
    wavs = [
        trim_wav(wavs[i], segment_info[i], win_len, trim_dither=trim_dither) for i in range(len(wavs))
    ] # wavs[wav1, wav2, wav3...] trim_wav => cut wav into 1s second
    rms = [w.norm(p=2).item() for w in wavs] # compute energy
    rms = list(map(lambda x: x if x > 0.01 else 1, rms)) # avoid devide very small value 0
    max_rms = max(rms) # find the max energy
    wavs = [wavs[i]*(max_rms/rms[i]) for i in range(len(rms))] # make wav1 wav2 to equal energy
    ratios = [r/sum(ratios) for r in ratios]
    scaled_wav = [wavs[i]*ratios[i] for i in range(len(wavs))]
    mix_wav = [sum(scaled_wav)]
    return mix_wav + scaled_wav, ratios

# trim wav by lenth and segment
def trim_wav(wav, seg_head, win_len=100, test_mode=False, trim_dither=False):
    length_idx = 1 if wav.size(0) == 1 else 0 # 1: waveform (channel,numsamples) 0: mfcc/fbank/spectrum (time, dim)
    win_len = DITHER_RANGE[length_idx] * 10 * win_len
    if seg_head != None and trim_dither:
        dither = DITHER_RANGE[length_idx]
        if seg_head <= dither:
            seg_head = RANDOM_FACTOR['int'](0, seg_head) if seg_head > 0 else 0
        else:
            seg_head = seg_head - RANDOM_FACTOR['int'](0, dither)

    if wav.size(length_idx) < win_len:
        wav_shape = wav.size()
        res_num = win_len - wav.size(length_idx) + 1
        if length_idx == 0:
            d = wav_shape[1]
            padding_shape = (res_num, d)
        else:
            d = wav_shape[0]
            padding_shape = (d, res_num)
        padding = torch.zeros(padding_shape)
        wav = torch.cat([wav,padding], dim=length_idx)
        
    #dither = (dither / 100) * 16000
    #win_len = int(win_len * 16000)

    #if seg_head != None: #TODO: the code is not safe ...
    #    if length_idx == 1:
    #        seg_head = (seg_head / 100)
    #    seg_head = seg_head*16000
    #else:
    #    if test_mode:
    #        seg_head = (wav.size(length_idx) - win_len) / 2
    #    else:
    #        seg_head = random.randint(0, wav.size(length_idx)-win_len)
    if seg_head == None:
        seg_head = random.randint(0, wav.size(length_idx)-win_len) if wav.size(length_idx)-win_len > 0 else 0
    seg_head = int(seg_head) if seg_head >= 0 else 0
    if seg_head + win_len > wav.size(length_idx):
        res = seg_head + win_len - wav.size(length_idx)
        seg_head = seg_head - res
        if seg_head < 0:
            seg_head = 0
    wav = wav[:,seg_head: seg_head + win_len]
    return wav


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
# detach_f: detach_functions, for self corruption the datalist will format as json object so detach_f will be json.dumps
# for none target corruption detach_f is lambda x: x
def make_corrupt_party(
        data, corrupt_list, corrupt_list_len, config, detach_f=lambda x: x.split(" ")[0]
    ):
    cdata = []
    ratios = []
    keywords = [] 
    segments = []
    speakers = []
    n = config.get('num_corrupt')
    if config.get('random_num', False):
        assert(n > 1)
        n = RANDOM_FACTOR['int'](1, n+1)
    sampler = config.get('sampler')
    assert (sampler in RANDOM_FACTOR.keys())
    sampler_config = config.get('sampler_config')
    if len(data) == 1: # random a ratio for data[0]
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)

    for i in (range(n)):
        one_corrupt = corrupt_list[np.random.randint(0, corrupt_list_len)]
        one_corrupt = detach_f(one_corrupt)
        if isinstance(one_corrupt, dict):
            keywords.append(one_corrupt['word_keyword'][0])
            if 'candidate_kw_idx' in one_corrupt:
                idx = one_corrupt['candidate_kw_idx']
                segments.append(one_corrupt['segment'][idx][0])
            elif 'segment' in one_corrupt:
                segments.append(one_corrupt['segment'][0])
            else:
                segments.append(None)
            cdata.append(one_corrupt['sph'])
            if 'speaker' in one_corrupt:
                speakers.append(one_corrupt['speaker'][0])
        elif isinstance(one_corrupt, str):
            keywords.append(None)
            segments.append(None)
            speakers.append(-1)
            cdata.append(one_corrupt)
        else:
            raise NotImplementedError("corrupt list error")
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        if sampler == 'beta':
            ratio = ratio.tolist()[0]
        #ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)
    return cdata, ratios, n, (keywords, segments, speakers)

# time shiftting
def time_shifting(wav, frame_length=400, hop_length=160, shift_type='mid', k=None):
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
    if k:
        save_wav(wav.view(1,-1), [k, idx_sample])
    return wav.view(1, -1) # NOTE: convert back to singal channel [1, num_sample]

#process raw json line
def process_raw(data):
    for sample in data:
        json_obj = sample['src']
        json_obj = json.loads(json_obj)
        one_sample = make_raw_sample_dict(json_obj)
        yield one_sample


# process speech feats:
# load input => [wav_augment] => [corruption / mixing ] => extract feature(mfcc,fbank..) => [spec augment]
# NOTE: when we do mixing, i.e. mix two target speech for MixUp or U-MixUp:
# 1. the keyword target will extended such as : {'target2': xx, 'target3': xx}
# 2. mixing ratio will alse append in training sample: {'ratio': ratio, 'ratio2': ratio2}
def process_speech_feats(data, config, max_len=3000):
    for sample in data:
        # load data
        input_data_type = config.get('data_type', 'raw')
        feats = sample['sph'].split(";") # some times we will add tow wav file in datalist, "wav1;wav2" => ['wav1','wav2']
        feats = [INPUT_DATA_LOADER[input_data_type](x) for x in feats]
        feats = [INPUT_DATA_LOADER['rm_sr'](x) for x in feats] if input_data_type == 'raw' else feats
        ratio = []
        num_clean = 1
        num_corrupt = n_scorrpute = n_ncorrupt = 0
        s_info = {}
        if config.get('self_corruption', False):
            self_corruption_config = config.get('self_corruption')
            corrupt_list = config['s_corrupt_list']
            corrupt_list_len = config['num_scorrupt_samples']
            s_feats, s_ratio, n_scorrpute, scrpt_keyword_info = make_corrupt_party(
                feats, corrupt_list, corrupt_list_len, self_corruption_config, json.loads
            ) # only retrun files name not wav
            for i, s in enumerate(s_feats):
                s_info.update({"wav{}".format(i): s, "keyword{}".format(i): scrpt_keyword_info[0][i]})
            num_corrupt += n_scorrpute
            num_clean += n_scorrpute
            s_feats = [INPUT_DATA_LOADER[input_data_type](x) for x in s_feats]
            s_feats = [
                INPUT_DATA_LOADER['rm_sr'](x) for x in s_feats
            ] if input_data_type == 'raw' else s_feats

            feats = feats + s_feats
            if self_corruption_config.get("shift_prob", False):
                shift_prob = self_corruption_config.get("shift_prob")
                shift_type = self_corruption_config.get("shift_type", 'mid')
                dice = np.random.randint(0,100) / 100
                if dice < shift_prob:
                    feats = [time_shifting(i, shift_type=shift_type) for i in feats]
            ratio += s_ratio
            #trim_win_size = max([f.size(1) for f in feats])
        #else:
        #    trim_win_size = None
        
        if config.get('none_target_corruption', False):
            corrupt_list = config['n_corrupt_list']
            corrupt_list_len = config['num_ncorrupt_samples']
            n_feats, n_ratio, n_ncorrupt, ncrpt_keyword_info = make_corrupt_party(
                feats, corrupt_list, corrupt_list_len, config.get('none_target_corruption')
            )
            n_feats = [INPUT_DATA_LOADER[input_data_type](x) for x in n_feats]
            n_feats = [
                INPUT_DATA_LOADER['rm_sr'](x) for x in n_feats
            ] if input_data_type == 'raw' else n_feats
            #trim_win_size = feats[0].size(0) if not trim_win_size else trim_win_size
            feats = feats + n_feats
            ratio += n_ratio
            num_corrupt += n_ncorrupt
        # END Loade input data
        if 'candidate_kw_idx' in sample:
            idx = sample['candidate_kw_idx'][0]
            segment = [copy.deepcopy(sample['segment'][idx][0])]
        elif 'segment' in sample:
            segment = [copy.deepcopy(sample['segment'])[0]]
        else:
            segment = [None]
        keyword = copy.deepcopy(sample['word_keyword'])

        if 'speaker' in sample:
            speaker = copy.deepcopy(sample['speaker'])
        else:
            speaker = [-1]

        # len(ratio) > 0 means that feats will do corrupt if len(feats)>1 but len(ratio)=0 means that only use tow 
        # single input without mix them
        if len(ratio) != 0: 
            if 'scrpt_keyword_info' in locals().keys(): # NOTE: use locals() to get local variable dict is not safe ... 
                keyword += scrpt_keyword_info[0] 
                segment += scrpt_keyword_info[1]
                speaker += scrpt_keyword_info[2] if speaker != [None] else [None]
            if 'ncrpt_keyword_info' in locals().keys():
                #keyword += ncrpt_keyword_info[0] 
                segment += ncrpt_keyword_info[1]
                ncrpt_dice = np.random.randint(0,5)
                if ncrpt_dice < 2:
                    c_feats = copy.deepcopy(feats[ncrpt_dice])
                    crpt_feats = copy.deepcopy(feats[-1])
                    tmp_feats = [c_feats, crpt_feats]
                    tmp_r = [ratio[ncrpt_dice], ratio[-1]]
                    tmp_segment = [segment[ncrpt_dice], segment[-1]]
                    #n_crupt_feats, _ = make_mix_wav(tmp_feats, tmp_r, tmp_segment, win_len=16000)
                    n_crupt_feats, _ = make_mix_wav(tmp_feats, tmp_r, tmp_segment, **config.get('trim_config', {}))
                    feats[ncrpt_dice] = n_crupt_feats[0]
                feats = feats[:-1]
                ratio = ratio[:-1]
                segment = segment[:-1]
            if len(feats) > 1:
                #feats, ratio = make_mix_wav(copy.deepcopy(feats), ratio, segment, win_len=16000)
                feats, ratio = make_mix_wav(copy.deepcopy(feats), ratio, segment, **config.get('trim_config', {}))
        elif config.get('trim_config', False):
            feats = [trim_wav(x, seg_head=segment[0], **config.get('trim_config', {})) for x in feats]
        # END Corruption
        #k = sample['key']
        #sc = s_info['wav0']
        #import os
        #sc = os.path.basename(sc)
        #zeros = torch.zeros_like(torch.rand(1,16000))
        #s_feats = [feats[0], zeros, feats[1], zeros, feats[2]]
        #c_feats = torch.cat(s_feats, dim=-1)
        #save_wav(c_feats, [k, sc])
        #exit()
        # Wav augment: add noise, do reverberate change speed ...
        if config.get('wav_augment', False):
            feats = [trim_wav(x, seg_head=segment[0], **config.get('trim_config', {})) for x in feats]
            feats = [wav_augment(f, config.get('wav_augment')) for f in feats]
        k = sample['key']
        feats_type = config.get('feats_type', 'fbank')
        feats_config = config.get('feats_config', FBANK_DEFAULT_SETTING)
        feats = [FEATS_EXTRACTOR[feats_type](f, **feats_config) for f in feats]


        # Splice Feature
        if config.get('splice_config'):
            splice_config = config.get('splice_config')
            feats = [splice_feats(f, **splice_config) for f in feats]

        # Subsample Feature
        if config.get('subsample_rate'):
            feats = [f[::config.get('subsample_rate')] for f in feats]

        feats = torch.cat([x.unsqueeze(0) for x in feats], dim=0)

        if num_corrupt == 0: #there are no any corruption just update clean speech feats
            sample.update({'speech': feats[0].clone()})
        else: # update mix speech feats
            mix_idx = torch.tensor([0])
            mix_feats = feats[mix_idx].squeeze(0)
            sample.update({'mixspeech': mix_feats.clone()})

        #if n_ncorrupt != 0: # there are none target corruption update augment speech
        #    aug_idx = torch.tensor([x for x in range(num_clean+1, feats.size(0))])
        #    aug_feats = feats[aug_idx].squeeze(0)
        #    sample.update({'augspeech': aug_feats.clone()})
        
        if n_scorrpute != 0:
            clean_idx = torch.tensor([x for x in range(1, num_clean+1)])
            clean_feats = feats[clean_idx]
            clean_feats = clean_feats.squeeze(0) if clean_feats.size(0) == 1 else clean_feats
            if config.get('concat_feats', False):
                sample.update({
                    'speech': clean_feats.clone(), 
                    'word_keyword': torch.tensor(keyword),
                    'ratio': torch.tensor(ratio),
                    'num_feats': torch.tensor([num_clean]),
                    'speaker': torch.tensor(speaker) 
                })
            else:
                for i, one_speech in enumerate(clean_feats):
                    one_keyword = keyword[i]
                    one_ratio = ratio[i]
                    one_speaker = speaker[i]
                    sample.update({
                        "speech{}".format(i+1): one_speech.clone(),
                        "word_keyword{}".format(i+1): torch.tensor([one_keyword]),
                        "ratio{}".format(i+1): torch.tensor([one_ratio]),
                        'speaker': torch.tensor([one_speaker])
                    })
        yield sample


# process alignment
def process_alignment(data, align_type="kaldi", max_len=1000):
    for sample in data:
        feats_len = sample['sph'].size(0)
        if align_type == 'kaldi':
            align = sample['align']
            align = kaldi_io.read_vec_int(align)
            align = torch.tensor(align)
            align = align[0:max_len]
        else:
            align = sample['segment']
            phone_label = sample['phone_label'].copy()
            align = convert_mfa_to_align(zip(phone_label, align))
            align = align[0:max_len]

        if feats_len > align.size(0):
            residual = feats_len - align.size(0)
            assert (residual < 3)  # if alignment is compute from mfa align
            # the length maybe missmatch because the time point is second
            one_padding = torch.ones(residual)
            align = torch.cat([align, one_padding], dim=0)

        sample.update({"align": align.clone()})
        yield sample

# process keyword: sampled keyword
def process_sampled_keyword(
    data,
    positive_prob=0.5,
    level='word',
):
    #TODO: optimize this code segment it look's like SHIT >._.<
    level = detach_level(level)

    for sample in data:
        dice = random.uniform(0, 1)
        neg_keyword_idx = -1
        keyword_idx = -1
        for prefix in level:
            label_key = prefix + "_label"
            keyword_key = prefix + "_keyword"
            neg_keywork_key = prefix + "_neg_keyword"

            if dice > positive_prob:  # negative sample
                neg_keyword_pool = sample[neg_keywork_key]
                if neg_keyword_idx == -1:
                    neg_keyword_idx = random.randint(0, len(neg_keyword_pool)-1)
                neg_keyword = neg_keyword_pool[neg_keyword_idx]
                sample.update({
                    keyword_key: neg_keyword.copy(),
                    label_key: [[230]],
                    'target': torch.tensor([0]),
                    "keyword_pos": torch.tensor([0])
                })
            else:
                #TODO: temple modify
                if keyword_idx == -1:
                    candidate_idx = sample['candidate_kw_idx']
                    num_keywrod = len(candidate_idx) 
                    keyword_idx = random.randint(0, num_keywrod-1)
                    keyword_idx = candidate_idx[keyword_idx]
                label = sample[label_key]
                keyword = label[keyword_idx].copy()
                sample.update({
                    label_key: label.copy(),
                    keyword_key: keyword.copy(),
                    "keyword_idx": keyword_idx,
                    'target': torch.tensor([1]),
                })
        yield sample


def process_fix_keyword(
    data,
    config=None
):
    for sample in data:
        yield sample


# process fix keyword
def process_fix_keyword_segment(
    data,
    positive_prob=1,
    level='word',
):
    #TODO: optimize this code fragment it look's like SHIT >._.<
    level = detach_level(level)

    for sample in data:
        if positive_prob >= 1:
            dice = 10
        else:
            dice = random.uniform(0, 1)
        neg_keyword_idx = -1
        keyword_idx = -1
        for prefix in level:
            keyword_key = prefix + "_keyword"
            neg_keyword_key = prefix + "_neg_keyword"

            if dice > positive_prob:  # negative sample
                neg_keyword_pool = sample[neg_keyword_key]
                if neg_keyword_idx == -1:
                    neg_keyword_idx = random.randint(
                        0,
                        len(neg_keyword_pool) - 1
                    )
                neg_keyword = neg_keyword_pool[neg_keyword_idx]
                sample.update({
                    keyword_key: neg_keyword.copy(), 
                    'target': torch.tensor([0])
                })
            else:
                if keyword_idx == -1:
                    candidate_idx = sample['candidate_kw_idx']
                    num_keywrod = len(candidate_idx) 
                    idx = random.randint(0, num_keywrod-1)
                    c_idx = candidate_idx[idx]
                    keyword_idx = idx
                keyword = sample[keyword_key][idx]
                sample.update({
                    keyword_key: [keyword],
                    "keyword_idx": c_idx,
                    'target': torch.tensor([1]),
                })
        yield sample

def process_test_set(
    data,
    level='word'
):
    level = detach_level(level)

    for sample in data:
        keyword_idx = sample['candidate_kw_idx'][0]
        if keyword_idx == -1:
            target_ = torch.tensor([0])
            keyword_idx_ = keyword_idx
        else:
            target_ = torch.tensor([1])
            keyword_idx_ = keyword_idx
        sample.update({'target': target_, 'keyword_idx': keyword_idx_})
        yield sample

def detach_level(level):
    if not isinstance(level, list):
        if ',' in level:
            level = level.split(",")
        else:
            level = [level]
    if any(
        l not in ['word', 'phone']
        for l in level
    ):
        raise NotImplementedError(
            "only support phone level and word level"
        )
    return level

# process special token keyword token and word token will be processed in this
# function
def process_special_token(
    data,
    level='word',
    token_config=None,
):

    #TODO: optimize this code fragment it look's like SHIT >._.<
    if "," in level:
        level = level.split(",")
    if not isinstance(level, list):
        level = [level]
    assert (x in ['word, phone'] for x in level)

    only_in_trans = token_config.get('only_in_trans', False)
    # special token in transcription and special token in keyword
    trans_token = token_config.get('trans_token', None)
    keyword_token = token_config.get('keyword_token', None)

    for sample in data:
        for prefix in level:
            label_key = prefix + "_label"
            keyword_key = prefix + "_keyword"
            label = sample[label_key].copy()
            keyword = sample[keyword_key].copy()
            if isinstance(keyword[0], list):
                keyword = keyword[0].copy()
            # keyword_idx = sample['keyword_idx']
            # inject special token into keyword and repalce it in transcription sequence
            # use keyword_tmp and token holder because sometimes the only keyword sequence
            # in transcription need special token

            if keyword_token:
                keyword_tmp = inject_special_token(keyword, **keyword_token)
                #TODO: sometimes special token only appeared in keyword sequence
                # not in transcription sequence ??
            
            if sample['target'] != 0:
                keyword_idx = sample['keyword_idx']
                label[keyword_idx] = keyword_tmp

            # inject speical token into transcription
            if trans_token:
                label = inject_special_token(label, **trans_token)
            
            if not only_in_trans:
                keyword = keyword_tmp.copy()
            
            sample.update({
                label_key: label.copy(),
                keyword_key: keyword.copy(),
            })
        yield sample


# all the label information are [[...], [...]] unfold them.
def process_list_data(dataset):
    keys = [
        'phone_label', 'word_label', 'phone_keyword', 'word_keyword', 'word_keyword2',
        'phone_neg_keyword', 'word_neg_keyword', 'word_label_aux', 'phone_label_aux',
        'num_feats', 'speaker'
    ]
    str_key = ['key', 'sph']
    word_keyword = ['word_keyword{}'.format(n) for n in range(1,10)]
    keys = keys + word_keyword
    speaker = ['speaker{}'.format(n) for n in range(1,10)]
    keys = keys + speaker
    for sample in dataset:
        for key, value in sample.items():
            if key in str_key:
                continue
            if isinstance(value, list):
                value = unfold_list(value)
            else:
                continue
            sample.update({key: torch.tensor(value)})
        yield sample


# such as ctc loss and rnn-t loss need speech length and target length
# but after make batch. these data will be append as the same length.
# so we compute length information before make batch
def make_length(dataset, length_key):
    # pre-define
    # keys can generate langth information
    # sph, label, keyword
    l_keys = [
        'mix','sph', 'sph1','wav1', 'wav2', 'word_label', 'phone_label', 'word_keyword', 'word_keyword2','phone_keyword', 
        'word_label_aux', 'phone_label_aux', 'speaker'
    ]
    l_keys = ['mix', 'sph', 'word_label', 'phone_label', 'speaker','word_keyword',  'word_label_aux', 'phone_label_aux']
    word_keyword = ['word_keyword{}'.format(n) for n in range(1,10)]
    speaker = ['speaker{}'.format(n) for n in range(1,10)]
    wav_idx = ['speech{}'.format(n) for n in range(1,10)]
    l_keys = l_keys + word_keyword + wav_idx + speaker
    #length_key = length_key.split(",")
    assert all(k in l_keys for k in length_key)
    for sample in dataset:
        length_info = {}
        for key in length_key:
            if key not in sample:
                continue
            new_key = "{}_len".format(key)
            length = sample[key].size(0)
            length_info.update({new_key: torch.tensor(length)})
        sample.update(length_info)
        yield sample

def random_trim_feats(dataset, win_len=200):
    for sample in dataset:
        sph = sample['sph'].clone()
        head = random.randint(sph.size(0)-win_len-1)
        new_sph = sph[head:win_len]
        trimed_feats = {'sph': new_sph}
        if 'wav1' in sample:
            trimed_feats.update({
                'wav1': sample['wav1'][head:win_len].clone()
            })
        if 'wav2' in sample:
            trimed_feats.update({
                'wav2': sample['wav2'][head:win_len].clone()
            })
        sample.update(new_sph)
        yield sample

# trim feats according to segment
def trim_feats(dataset, dither=10, win_len=100):
    if isinstance(dither, str):
        assert dither == "fix"
        dither = 5
        fix = True
    else:
        fix = False

    for sample in dataset:
        if 'keyword_idx' not in sample:
            candidate_idx = sample['candidate_kw_idx']
            num_keywrod = len(candidate_idx) 
            keyword_idx = random.randint(0, num_keywrod-1)
            keyword_idx = candidate_idx[keyword_idx]
        else:
            keyword_idx = sample['keyword_idx']
        trim_segment = sample['segment'][keyword_idx]
        head = trim_segment[0]
        if fix:
            head = head - 5 if head > 5 else head
        else:
            head = head - random.randint(0, dither-1) if head > dither else \
                head - random.randint(0, head)
        feats = sample['sph'].clone()
        ratio = sample.get('ratio', None)
        if ratio != None:
            wav1 = sample['wav1'].clone()
            wav2 = sample['wav2'].clone()
            wav1 = wav1[head:head + win_len]
            wav2 = wav2[head:head + win_len]
        feats = feats[head:head + win_len]
        if feats.size(0) != win_len:
            d = feats.size(1)
            res = win_len - feats.size(0)
            padding = torch.ones_like(torch.rand(res, d))
            padding = padding * feats[-1]
            feats = torch.cat([feats, padding], dim=0)
            if ratio != None:
                wav1 = torch.cat([wav1, padding], dim=0)
                wav2 = torch.cat([wav2, padding], dim=0)
        if ratio != None:
            sample.update({
                'sph': feats,
                'wav1': wav1,
                'wav2': wav2
            })
        else:
            sample.update({'sph':feats})
        yield sample


# concat into one batch
def concat_tensor(
    data_list,
    seq_padding=False,
    padding_value=0
):
    if seq_padding:
        tensor = pad_sequence(data_list, batch_first=True, padding_value=padding_value)
    else:
        tensor = torch.cat([x.unsqueeze(0) for x in data_list], dim=0)
    return tensor


# fetch keys
def fetch_tensor(
    data,
    fetch_key,
):
    seq_keys = [
        'key', 'word_label', 'phone_label', 'align', 'word_keyword', 
        'phone_keyword', 'phone_neg_keyword', 'word_neg_keyword',
        'mixspeech', 'augspeech', 'speech', 'ratio', 'speaker'
    ]
    word_lable = ['word_keyword{}'.format(x) for x in range(1, 10)]
    speaker_lable = ['speaker{}'.format(x) for x in range(1, 10)]
    speech_lable = ['speech{}'.format(x) for x in range(1, 10)]
    ratio_lable = ['ratio{}'.format(x) for x in range(1, 10)]
    seq_keys = seq_keys + word_lable + speech_lable + ratio_lable + speaker_lable
    for sample in data:
        if fetch_key[0] == 'key':
            sort_key = fetch_key[1]
        else:
            sort_key = fetch_key[0]
        index = torch.tensor([x[sort_key].size(0) for x in sample])
        index = torch.argsort(index, descending=True)
        return_feats = []
        for k in fetch_key:
            if k in seq_keys:
                seq_padding = True
            else:
                seq_padding = False
            if (k == 'key') or (k == 'tag'):
                return_feats.append([sample[i][k] for i in index])
            else:
                if any(
                    t in k
                    for t in ['label']
                ):
                    padding_value = -1
                else:
                    padding_value = 0
                return_feats.append(
                    concat_tensor(
                        [sample[i][k] for i in index], 
                        seq_padding=seq_padding,
                        padding_value=padding_value)
                )
        yield tuple(return_feats)


# make batch
def make_batch(data, batch_size=256):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf

# check label data: convert label id to sym char
def check_label_data(data, sym_tabel):
    for sample in data:
        phone_lable = sample['phone_label'] 
        char_label = [sym_tabel[x.item()] for x in phone_lable]
        uttid = sample['key']
        if "$0" in char_label:
            print (uttid, char_label, sample['target'])
        
        yield sample

#################################################
### the following def. are previous functions ###
#################################################
# read feats only
def read_mat_scp(data):
    for sample in data:
        key = sample['key']
        feats_ark = sample['feats_ark']
        feats = kaldi_io.read_mat(feats_ark)
        example = dict(key=key, feats=feats)
        yield example


# splice feats with context
def splice_feats(feats, left_context, right_context, seq=True):
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


def tensor2str(t):
    if isinstance(t, torch.Tensor):
        t = t.numpy()
    t = list(t)
    t = list(map(lambda x: str(x), t))
    return t


def read_mat_with_segment(data):
    for sample in data:
        json_line = sample['src']
        d = json.loads(json_line)
        key = d['key']
        feats_ark = d['feats_ark']
        label, head, tail, sil_tail = d['label'].split(" ")
        if int(label) == 1:
            continue
        feats = copy.deepcopy(kaldi_io.read_mat(feats_ark))
        feats = torch.tensor(feats)
        label = torch.tensor([int(label)])
        head = int(head)
        tail = int(tail)
        sil_tail = int(sil_tail)
        if sil_tail > 500:
            sil_tail = 200
        sil_dice = np.random.randint(0, 300)
        if sil_dice == 2:
            if sil_tail < 50:
                feats = feats[0:sil_tail]
                padding = feats[-1].clone().unsqueeze(0)
                num_padding = 100 - feats.size(0)
                padding = padding.repeat(num_padding, 1)
                feats = torch.cat([feats, padding], dim=0)
            else:
                feats = feats[0:100]
            label = label * 0
        else:
            if int(label) == 1 and tail - head > 100:
                head = np.random.randint(head, tail - 100)

            if head > 10:
                head = head - np.random.randint(0, 10)
            feats = feats[head:head + 100]

        example = dict(key=key, feats=feats, label=label.to(torch.long))
        yield example


# read mat[feats.ark] and alignment[ali.scp]
def read_mat_with_ali(data, double_label=None):
    for sample in data:
        json_line = sample['src']
        d = json.loads(json_line)
        key = d['key']
        feats_ark = d['feats_ark']
        feats = copy.deepcopy(kaldi_io.read_mat(feats_ark))
        feats = torch.tensor(feats)
        frames = feats.size(0)
        if double_label:
            label1_file = d['label1']
            label2_file = d['label2']
            if " " not in label1_file:
                label1 = kaldi_io.read_vec_int(label1_file)
                slabel1 = tensor2str(label1.copy())
                slabel1 = " ".join(slabel1)
            else:
                label1 = sym2int(label1_file.split(" "))
                slabel1 = copy.deepcopy(label_file)
            if " " not in label2_file:
                label2 = kaldi_io.read_vec_int(label2_file)
                slabel2 = list(map(lambda x: str(x), list(label2)))
                slabel2 = " ".join(slabel2)
            else:
                label2 = sym2int(label2_file.split(" "))
                slabel2 = copy.deepcopy(label2_file)
            label1 = torch.tensor(label1, dtype=torch.long)
            label2 = torch.tensor(label2, dtype=torch.long)
            example = dict(key=key,
                           feats=feats,
                           feats_len=torch.tensor([frames], dtype=torch.long),
                           label1=label1,
                           label2=label2,
                           slabel1=slabel1,
                           slabel2=slabel2)
            yield example
        else:
            label_file = d['label']
            if " " not in label_file:
                label = kaldi_io.read_vec_int(label_file)
                slabel = tensor2str(label.copy())
            else:
                label = sym2int(label_file.split(" "))
                slabel = copy.deepcopy(label_file)
            label = torch.tensor(label, dtype=torch.long)
            example = dict(key=key,
                           feats=feats,
                           feats_len=torch.tensor([frames], dtype=torch.long),
                           label=label,
                           slabel=slabel)
            yield example


# random select keyword
def make_keyword(label, max_kw_len=20):
    if isinstance(label, torch.Tensor):
        label = list(label.numpy())
    label = list(label).copy()
    phn_p = label[0]
    lphn_id = 0  # init local phone id
    lphn_pos = {}  # local phone id position lphn_id:[start frame, end frame]
    lphn_pos[lphn_id] = [0]
    l2g_map = {lphn_id: phn_p}  # local phone id to global phone id map
    for index, phn in enumerate(label):
        if phn == phn_p:
            continue
        else:
            lphn_pos[lphn_id + 1] = [index]
            lphn_pos[lphn_id].append(index - 1)
            l2g_map[lphn_id + 1] = phn
            lphn_id += 1
            phn_p = phn

    lphn_pos[lphn_id].append(len(label))
    num_phones = len(l2g_map)
    if num_phones > 2:
        kw_len = np.random.randint(2, num_phones)
        kw_head = np.random.randint(1, kw_len)
        keyword = list(l2g_map.keys())
        keyword = keyword[kw_head:kw_head + kw_len]
        keyword = keyword[:max_kw_len]
        kw_ali_head = lphn_pos[keyword[0]][0]
        kw_ali_tail = lphn_pos[keyword[-1]][1]
        kw_ali = [
            1 if x >= kw_ali_head and x <= kw_ali_tail else 0
            for x in range(len(label))
        ]
    else:
        keyword = list(l2g_map.keys())
        kw_ali = [1 for x in range(len(label))]

    keyword = [l2g_map[k] for k in keyword]
    kw_len = len(keyword)
    return torch.tensor(keyword, dtype=torch.long), \
        torch.tensor(kw_len, dtype=torch.long), torch.tensor(kw_ali)


# sample keyword from alignment
def sample_keyword(data,
                   max_kw_len,
                   win_size=None,
                   p_postive=0.5,
                   double_label=False):
    for sample in data:
        if double_label:
            dice = np.random.randint(0, 10)
            if dice // 2 == 0:
                label = sample['label1']
            else:
                label = sample['label2']
        else:
            label = sample['label']

        feats = sample['feats']
        frames = feats.size(0)
        if isinstance(win_size, int):
            sample_head = frames - win_size
            sample_head = np.random.randint(0, sample_head)
        else:
            sample_head = 0
            win_size = frames
        feats = feats[sample_head:sample_head + win_size]
        label = label[sample_head:sample_head + win_size]
        pos_neg_dice = np.random.randint(0, 100)

        if pos_neg_dice < p_postive * 100:
            keyword, kw_len, kw_ali = make_keyword(label, max_kw_len)
            kw_label = torch.tensor([1])
        else:
            keyword = None
            kw_label = torch.tensor([0])
            kw_len = None
            kw_ali = torch.zeros(frames)

        sample['feats'] = feats
        sample['feats_len'] = win_size
        sample['label'] = label
        sample['keyword'] = keyword
        sample['kw_label'] = kw_label
        sample['kw_ali'] = kw_ali
        sample['kw_len'] = kw_len
        yield sample


# google keyword sample
def make_google_pos(unk_id, eow_id, label):
    zeros = torch.zeros_like(label)
    eow_index = torch.where(label == eow_id, label, zeros)
    eow_index = torch.nonzero(eow_index)
    eow_index = eow_index.view(-1)
    num_w = eow_index.size(0)
    insert_index = np.random.randint(0, num_w)
    if insert_index == 0:
        key_i_head = 0
        key_i_tail = int(eow_index[0])
    else:
        key_i_head = int(eow_index[insert_index - 1])
        key_i_tail = int(eow_index[insert_index])
    kw_len = key_i_tail - key_i_head - 1
    key_word = label[key_i_head + 1:key_i_tail]
    if unk_id in key_word:
        key_word = None
        kw_len = 0
        return label, key_word, kw_len
    else:
        label = label.index_put([torch.LongTensor([key_i_tail])],
                                torch.LongTensor([eow_id + 1]))
        return label, key_word, kw_len


# make google_kw training sample: insert eokw into label
def sample_google_kw(data, unk_id, eow_id, p_positive=0.5, double_label=False):
    for sample in data:
        if double_label:
            dice = np.random.randint(0, 10)
            if dice // 2 == 0:
                label = sample['label1']
            else:
                label = sample['label2']
        else:
            label = sample['label']
        pos_or_neg = np.random.randint(100)
        if pos_or_neg < p_positive * 100:  # pos
            label, key_word, kw_len = make_google_pos(unk_id, eow_id, label)
            sample['label'] = label
            sample['kw_label'] = torch.tensor([1])
            sample['kw_len'] = torch.tensor([kw_len], dtype=torch.long)
            sample['keyword'] = key_word
        else:  # neg
            sample['label'] = label
            sample['keyword'] = None
            sample['kw_len'] = torch.tensor([0], dtype=torch.long)
            sample['kw_label'] = torch.tensor([0])
        yield sample


def get_phone_ali(data, double_label=False):
    #TODO: finish
    for sample in data:
        if double_label:
            dice = np.random.randint(0, 10)
            if dice // 2 == 0:
                label = sample['label1']
            else:
                label = sample['label2']
        else:
            label = sample['label']


# append negative sample into data
def append_neg(data, neg_pool, neg_pool_size, max_kws_len=None):
    for sample in data:
        if sample['keyword'] != None:
            yield sample
        utt_label = sample['slabel']
        sone_neg = copy.deepcopy(utt_label)
        while sone_neg == utt_label:
            neg_idx = np.random.randint(0, neg_pool_size)
            one_neg = neg_pool[neg_idx]
            if max_kws_len:
                one_neg = one_neg[:max_kws_len]
            sone_neg = tensor2str(one_neg)

        one_neg = torch.tensor(one_neg, dtype=torch.long)
        one_neg_len = torch.tensor([len(one_neg)], dtype=torch.long)
        sample['keyword'] = one_neg
        sample['kw_len'] = one_neg_len
        yield sample


def empty(data):
    for sample in data:
        yield sample
