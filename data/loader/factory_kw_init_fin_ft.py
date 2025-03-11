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
import data.loader.data_utils_kw_init_fin_ft as utils

from typing_extensions import Tuple, Dict, Iterator, Any, Optional
from local.zeus_error import DataLoaderError

# Pre-Defined None-Tensor Key & CTC Tag
NONE_TENSOR_KEY = [
    'wav', 'key', 'sph', 'corruption_material', 'segment', 'segment_idx',
    'n_scorrupt', 'n_ncorrupt', 'num_corrupt', 'rirs', 'neg_candidate',
    'corrupt', 'self_corruption', 'none_target_corruption', 'nframes', 'ref1', 'ref2', 'ref3', 'ref0',
    'self_crpt_ratios', 'noise_crpt_ratios', 'self_crpt_material', 'noise_crpt_material', 's_name'
]
#    'self_crpt_ratios', 'noise_crpt_ratios', 'self_crpt_material', 'noise_crpt_material', 's_name', 'aux_lexicon'
CTC_KEY = [
    'label', 'crpt_label', 'phn_label', 'bpe_label', 'c_phn_label', 'c_bpe_label',
    'c_bpe_label0','c_bpe_label', 'c_bpe_label1', 'c_bpe_label2',
    'c_phn_label0', 'c_phn_label1','c_phn_label2', 'neg_label', 'fifo_label',
    'bpe_label0', 'bpe_label1', 'bpe_label2',
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
    'rm_sr': lambda x: x[0], 'copy': copy.deepcopy, 'empty': lambda x: x,
    'read_sr': lambda x: x[1]
}

# mfcc fbank spectrum factory
FEATS_EXTRACTOR = {
    'mfcc': kaldi.mfcc, 'fbank': kaldi.fbank, 'spectrum': torchaudio.functional.spectrogram, 
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

finetune_data_map = {'target_pos': 0, 'target_neg': 1, 'base_neg': 2}

# process raw json line
# data list is aranged in json format, in this function convert json into dict
# NOTE:  egs_format is a test feature i.e. read data from egs file just same like kaldi
# But egs_format didn't boost the training speed yet. just keep it and waiting for tuning
def process_raw(data: Iterator) -> Iterator[Dict[Any, Any]]:
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
        if 'kw_lexicon' in sample:
            kw_lexicon = sample['kw_lexicon']
            one_sample.update({'kw_lexicon': kw_lexicon})
        if 'finetune_data' in one_sample:
            finetune_data = one_sample['finetune_data']
            one_sample.update({'finetune_data': finetune_data_map[finetune_data]})

        epoch = sample['epoch']
        one_sample.update({'epoch': epoch})
        yield one_sample

# make corruption
# mix wav: 
#    - self corrution means mix two target speech: such as  speech + speech
#    - none_target corruption means target speech with noise: such as keyword1 speech + none target inteferance
# NOTE: in this function, waveform are not mixed !!!  Just extract mix mmaterials !!! e.g.:
# corruption_material:{1: speech_1 FILE, 2:  speech_2 FILE, 3: niose speech FILE}
# corruption ratios: [0.1, 0.6, 0.5]    
# the function is process_speech_feats:mix_wav will employ corruption material and corruption ratios to make
# the real mix waveform!!!!
def process_corruption(data: Iterator, config: Dict[Any, Any]) -> Iterator[Dict[Any, Any]]:
    for sample in data:
        s_ratios = n_ratios = list()
        s_corruption_material = n_corruption_material = dict()
        num_corrupt = n_scorrupt = n_ncorrupt = 0
        if config.get('self_corruption', False): # make self corruption materials; s_* means self_*
            assert 'self_corruption' in sample
            corrupt_list = sample['self_corruption']
            corrupt_list_len = len(corrupt_list) 
            s_ratios, s_corruption_material, n_scorrupt, max_scorrupt = utils.make_corrupt_party(
                corrupt_list, corrupt_list_len, config['self_corruption'], 'self', 
            )
            num_corrupt += n_scorrupt

            sample.update({
                'self_crpt_ratios': s_ratios,
                'self_crpt_material': s_corruption_material,
                'n_scorrupt': n_scorrupt,
                'n_max_scorrupt': max_scorrupt
            })
        
        if config.get('none_target_corruption', False): # make none target corruption materials
            assert 'none_target_corruption' in sample
            corrupt_list = sample['none_target_corruption']
            corrupt_list_len = len(corrupt_list)
            n_ratios, n_corruption_material, n_ncorrupt, _ = utils.make_corrupt_party(
                corrupt_list, corrupt_list_len, config['none_target_corruption'], 'noise'
            )
            num_corrupt += n_ncorrupt

            sample.update({
                'noise_crpt_ratios': n_ratios,
                'noise_crpt_material': n_corruption_material,
                'n_ncorrupt': n_ncorrupt
            })

        sample.update({
            'num_corrupt': num_corrupt
        })

        yield sample

# process speech feats
# load wav -> corrupt wav -> destroy a positive sample to negative (made for FA) -> extract fbank
# NOTE: this function can support load kaldi ark feats, torch pt file and read wavefrom from raw wav file
# NOTE: But kaldi feat, torch pt have not been verified in training process be carefull that.
def process_speech_feats(data: Iterator[Dict], config: Dict[Any, Any]) -> Iterator[Dict]:
    for sample in data:
        input_data_type = config.get('data_type', 'raw') # feats type: raw=>waveform kaidl: kaldi ark, pt: torch.pt
        if (input_data_type != 'raw') and ('corruption_material' in sample): # corruption only support performed on waveform
            raise NotImplementedError("Only support corruption on waveforme")
        speech_feats = [sample['sph']]

        if 'self_crpt_material' in sample:
            self_crpt_material = sample['self_crpt_material']
            self_crpt_feats = [self_crpt_material[x]['sph'] for x in self_crpt_material.keys()] 
            self_crpt_ratios = sample['self_crpt_ratios']
            speech_feats = speech_feats + self_crpt_feats
        else:
            self_crpt_ratios = [1]
        
        if 'noise_crpt_material' in sample:
            noise_crpt_material = sample['noise_crpt_material']
            noise_feats = [noise_crpt_material[x]['sph'] for x in noise_crpt_material.keys()] 
            noise_crpt_ratios = sample['noise_crpt_ratios']
        else:
            noise_feats = list()
            noise_crpt_ratios = list()

        # detach speech feats and noise feats
        speech_feats = [INPUT_DATA_LOADER[input_data_type](x) for x in speech_feats]
        sample_rate = INPUT_DATA_LOADER['read_sr'](speech_feats[0])  # TODO: this is a temp code 
        speech_feats = [INPUT_DATA_LOADER['rm_sr'](x) for x in speech_feats] if input_data_type == 'raw' else speech_feats
        noise_feats = [INPUT_DATA_LOADER[input_data_type](x) for x in noise_feats]
        noise_feats = [INPUT_DATA_LOADER['rm_sr'](x) for x in noise_feats] if input_data_type == 'raw' else noise_feats

        # Wav augment: volume & speed change
        if config.get('wav_augment', False):
            wav_augment_config = config.get('wav_augment')
            speech_feats = [utils.wav_augment(f, wav_augment_config, sample_rate) for f in speech_feats]
            

        # reverb_aug: reverbration 
        if sample.get('rirs', None):
            rirs_config = config.get('rirs')
            assert ('rirs' in sample)
            rirs_src = sample['rirs']
            speech_feats = [utils.reverb_aug(f, rirs_config, rirs_src) for f in speech_feats] 

        if sample.get('num_corrupt', 0) > 0:
            mix_config = sample.get('mix_config', {})
            feats = utils.make_mix_wav(speech_feats, self_crpt_ratios, noise_feats, noise_crpt_ratios, **mix_config)
        else:
            feats = speech_feats
        sample.update({'mix_wav': feats[0]}) 
        if config.get('return_raw_wav', False):
            for i, data in enumerate(speech_feats):
                sample_len = data.size(1)
                dur = 16000 * 4 
                sample_head = random.randint(0, sample_len-dur-1) if sample_len > dur else 0
                raw_wav = copy.deepcopy(speech_feats[i][:, sample_head:sample_head+dur])
                sample.update({'raw_wav{}'.format(i): raw_wav.squeeze(0)})  

        # Extract feature: MFCC / FBANK 
        feats_type = config.get('feats_type', 'fbank')
        feats_config = config.get('feats_config', FBANK_DEFAULT_SETTING)
        feats = [FEATS_EXTRACTOR[feats_type](f, **feats_config) for f in feats]

        # Spec Augment: time & freq mask
        if config.get('spec_augment', False):
            spec_augment_config = config.get('spec_augment')
            feats = [utils.spec_augment(f, spec_augment_config) for f in feats]

        # Splice Feature: add context
        if config.get('splice_config'):
            splice_config = config.get('splice_config')
            feats = [utils.splice_feats(f, **splice_config) for f in feats]

        # Subsample Feature: skip frame
        if config.get('subsample_rate'):
            feats = [f[::config.get('subsample_rate')] for f in feats]

        # Load feats into torch Tensor
        start_idx = 0
        if 'self_crpt_ratios' in sample:
            mix_feats = feats[0]
            sample.update({"mixspeech": mix_feats})
        else: # if no corruption meterail the 1th feats is clean feats
            sample.update({"speech": feats[0]})
        
        # keep clean feats e.g. mix_wav = wav1 + wav2 the following code will 
        # concat wav1, wav2 into one matrix and load to {speech: [wav1; wav2]}
        if sample.get("n_scorrupt", 0) > 0:
            start_idx = 1 # the first one is mix speech
            clean_feats = feats[start_idx: start_idx+sample['n_scorrupt']+1]
            ratios = sample['self_crpt_ratios']
            #TODO: consider about add noise augment ratios
            ratios = ratios[0:sample['n_scorrupt']+1]
            clean_feats = torch.cat([x.unsqueeze(0) for x in clean_feats], dim=0)
            start_idx = sample['n_scorrupt'] + 1
            sample.update({"speech": clean_feats[0]}) # here only keep the first wav
            sample.update({"ratios": ratios})
        yield sample 


def process_text_feats(data: Iterator[Dict]) -> Iterator[Dict]:
    for sample in data:
        if 'self_crpt_material' in sample:
            #c_keywords, c_labels, c_phn_labels, c_segment_labels, c_bpe_labels, kw_candidates, b_kw_candidates = utils.detach_corruption(
            c_keywords, c_labels, c_phn_labels, c_bpe_labels, kw_candidates, b_kw_candidates = utils.detach_corruption(
                sample['self_crpt_material']
            )
            if len(c_keywords) != 0:
                mix_keyword = copy.deepcopy(sample['keyword'])
                mix_keyword.append(copy.deepcopy(c_keywords))
                sample.update({
                    'keyword{}'.format(i+1): c_keywords[i] for i in range(len(c_keywords))
                })
                sample.update({'mix_keyword': mix_keyword})

            if len(c_labels) != 0:
                mix_label = copy.deepcopy(sample['label'])
                mix_label.append(copy.deepcopy(c_labels))
                sample.update({
                    'label{}'.format(i+1): c_labels[i] for i in range(len(c_labels))
                })
                sample.update({'mix_label': mix_label})
             
            if len(c_phn_labels) != 0:
                mix_phn_label = copy.deepcopy(sample['phn_label'])
                mix_phn_label.append(copy.deepcopy(c_phn_labels))
                sample.update({
                    'phn_label{}'.format(i+1): c_phn_labels[i] for i in range(len(c_phn_labels))
                })
                sample.update({'mix_phn_label': mix_phn_label})
           
            #if len(c_segment_labels) != 0:
            #    mix_phn_label = copy.deepcopy(sample['phn_label'])
            #    mix_phn_label.append(copy.deepcopy(c_phn_labels))
            #    sample.update({
            #        'phn_label{}'.format(i+1): c_phn_labels[i] for i in range(len(c_phn_labels))
            #    })

            if len(c_bpe_labels) != 0:
                mix_bpe_label = copy.deepcopy(sample['bpe_label'])
                mix_bpe_label.append(copy.deepcopy(c_bpe_labels))
                sample.update({
                    'bpe_label{}'.format(i+1): c_bpe_labels[i] for i in range(len(c_bpe_labels))
                })
                sample.update({'mix_bpe_label': mix_bpe_label})

            if len(kw_candidates) != 0:
                sample.update({
                    'kw_candidate{}'.format(i+1): kw_candidates[i] for i in range(len(kw_candidates))
                }) 
            if len(b_kw_candidates) != 0:
                sample.update({
                    'b_kw_candidate{}'.format(i+1): b_kw_candidates[i] for i in range(len(b_kw_candidates))
                }) 
            
        yield sample


def process_conditional_chain_label(data: Iterator[Dict], special_token: Dict) -> Iterator[Dict]:
    for sample in data:
        labels = [sample['bpe_label']] + \
            [sample['bpe_label{}'.format(i+1)] for i in range(sample['n_scorrupt'])]
        n_speech = sample['n_scorrupt'] + 1
        speech_idx = [i for i in range(n_speech)]
        condition = list()
        n_condition = random.randint(0, n_speech-1)
        label_mask = [0 for _ in range(sample['n_max_scorrupt']+1)]
        if n_condition == 0: # there are not any speech component has been extracted
            condition = [0] if 'sos' not in special_token else [special_token['sos']]
            for i in range(len(labels)):
                labels[i].append(special_token['sc'])
        elif n_condition == n_speech-1: # only one sentence wait for recognize
            label_idx = random.sample(speech_idx, 1)
            for i, l in enumerate(labels):
                if i != label_idx[0]:
                    label_mask[i] = 1
                    condition.append(l)
                    condition.append(special_token['sc'])
            labels[label_idx[0]].append(special_token['stop'])
            #target_label = labels[label_idx[0]] + [special_token['stop']]
            #labels = [target_label for i in range(n_speech)]
        else: # at least two speech wait for recognize
            label_idx = random.sample(speech_idx, n_condition)
            for i, l in enumerate(labels):
                if i not in label_idx:
                    label_mask[i] = 1
                    condition.append(l)
                    condition.append(special_token['sc'])
                else:
                    labels[i].append(special_token['sc'])
        sample.update({
            'condition': condition,
            'label_mask': label_mask
        }) 
        for i, l in enumerate(labels):
            sample.update({'bpe_label{}'.format(i): l})
        yield sample

            
# Process text feats, mainly deal with segment:
# e.g. in process_speech_feats wav has been trimed by segment, and the text label will 
# be cutted in this function acorrding to segment also.
def process_text_feats_dump(data, neg_token=None, sc_token=None):
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
            c_keyword, c_label, c_phn_label, c_bpe_label = utils.detach_corruption(sample['corruption_material'], c_segment_idx)

            if len(c_keyword) != 0:
                sample.update({"mix_keyword": sample['word_keyword']+utils.unfold_list(c_keyword)})
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
        data: Iterator[Dict], positive_prob: float=0.5, neg_len: int = None, special_token: Dict = {}, min_keyword_len: int=2, max_keyword_len: int=6
):
    # TEXT_SPEC_TOKEN = {'sos','eos','sok', 'eok', 'unk'}
    # sos: start of setence, eos: end of setence, sok: start of keyword, eok, end of keyword, unk: unknow token
    TEXT_SPEC_TOKEN.update(special_token)
    for sample in data:
        new_phn_label = copy.deepcopy(sample['phn_label'])
        #new_segment_label = copy.deepcopy(sample['segment_label'])
        new_bpe_label = copy.deepcopy(sample['bpe_label']) if 'bpe_label' in sample else None
        bpe_candidate = copy.deepcopy(sample['b_kw_candidate']) if 'b_kw_candidate' in sample else None
        #num_pre_sample = 5
        finetune_data = copy.deepcopy(sample['finetune_data'])
        kw_lexicon = copy.deepcopy(sample['kw_lexicon'])
        corrupt_label = None if 'mix_phn_label' not in sample else sample['mix_phn_label']
        kw, kw_pos, kw_length, pos, target = utils.make_keyword(
            candidate_seq=new_phn_label, kw_lexicon=kw_lexicon,
            finetune_data=finetune_data
        )
        kw, new_phn_label, new_bpe_label, kw_pos = utils.inject_special_token(
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
        permuates = utils.permuate_labels(labels, sc_token)
        for idx, per in enumerate(permuates):
            sample.update({"per{}".format(idx): per})
        yield sample

# process permuate label
def process_permuate_label_with_keyword(
        data, sc_token=None, positive_prob=0.5, neg_len=5, special_token=None
    ):
    for sample in data:
        labels = []
        bpe_labels = []
        b_kw_candidates = []
        kw_candidates = []
        labels.append(copy.deepcopy(sample['phn_label']))
        bpe_labels.append(copy.deepcopy(sample['bpe_label']))
        b_kw_candidates.append(sample['b_kw_candidate'])
        kw_candidates.append(sample['kw_candidate'])
        for x in range(1, sample['n_scorrupt']+1):
            labels.append(copy.deepcopy(sample['phn_label{}'.format(x)]))
        for x in range(1, sample['n_scorrupt']+1):
            bpe_labels.append(copy.deepcopy(sample['bpe_label{}'.format(x)]))
        for x in range(1, sample['n_scorrupt']+1):
            b_kw_candidates.append(copy.deepcopy(sample['b_kw_candidate{}'.format(x)]))
        for x in range(1, sample['n_scorrupt']+1):
            kw_candidates.append(copy.deepcopy(sample['kw_candidate{}'.format(x)]))

        candidate_label_idx = random.sample([x for x in range(len(labels))], 1)[0]
        candidate_label = labels[candidate_label_idx]
        candidate_bpe_label =  bpe_labels[candidate_label_idx]

        keyword, keyword_pos, keyword_len, pos, target = utils.make_keyword(
            candidate_label, sample['neg_candidate'], positive_prob, neg_len, kw_candidates[candidate_label_idx]
        )
        keyword, candidate_label, candidate_bpe_label, keyword_pos = utils.inject_special_token(
            keyword=keyword, keyword_length=keyword_len, 
            label=candidate_label, positive=pos, keyword_pos=keyword_pos,
            special_token=special_token, bpe_label=candidate_bpe_label, bpe_candidate=b_kw_candidates[candidate_label_idx]
        )
        if 'with_per_label' in special_token:
            labels[candidate_label_idx] = candidate_label
            bpe_labels[candidate_label_idx] = candidate_bpe_label

        if 'with_trans' in special_token:
            if special_token['with_trans']:
                sample.update({'bpe_label': candidate_bpe_label,'phn_label': candidate_label})

        permuates = utils.permuate_labels(labels, sc_token, max_mix_num=sample['n_max_scorrupt']+1)
        bpe_permuates = utils.permuate_labels(bpe_labels, sc_token, max_mix_num=sample['n_max_scorrupt']+1)

        for idx, per in enumerate(permuates):
            sample.update({"per{}".format(idx): per})
        for idx, per in enumerate(bpe_permuates):
            sample.update({"bpe_per{}".format(idx): bpe_permuates})
        #sample.update({'s_name': save_name})
        #with open('{}.txt'.format(sample['s_name']), 'w') as wf:
        #    x = permuates[0]
        #    wf.write(str(x)+"\n")
        #    x = bpe_permuates[0]
        #    wf.write(str(x)+"\n")
        #    wf.write(str(keyword)+"\n")
        #    wf.write(str(candidate_label)+"\n")
        #    wf.write(str(candidate_bpe_label)+"\n")
        sample.update({
            'keyword': keyword, 'target': target, 'bpe_label': candidate_bpe_label,
            'phn_label': candidate_label
        })
        yield sample

# process sot label, i.e. add sc token into label
def process_sot_label(data, special_token=None):
    for sample in data:
        new_label = copy.deepcopy(sample['bpe_label'])
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
            one_kw, _, kw_length, _, _ = utils.make_keyword(sample, positive_prob=1.2, neg_len=None)
            one_kw, _, _, _ = utils.inject_special_token(one_kw, kw_length, special_token=special_token)
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
            one_kw, _, kw_length, _, _ = utils.make_keyword(one_crpt, positive_prob=1.2, neg_len=None)
            one_kw, _, _, _ = utils.inject_special_token(one_kw, kw_length, special_token=special_token)
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


# process fix keyword from segment
def process_fix_keyword(data, special_token={}):
    for sample in data:
        if len(special_token) != 0:
            kw = sample['keyword']
            label = sample['label']
            kw, label, _ = utils.inject_special_token(
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
                value = utils.unfold_list(value)
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
            try:
                return_feats.append(
                    utils.concat_tensor(
                        [sample[i][k] for i in index], seq_padding=seq_padding, padding_value=padding_value
                    )
                )
            except:
                raise DataLoaderError(
                    'Erorr when fetch_tensor key: {}'.format(k), pos='fetch_tensor: 571'
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
