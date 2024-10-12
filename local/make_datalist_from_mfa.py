import os
import argparse
import json
import sys
sys.path.append('./')
from typing import Optional, Dict, Tuple
from local.utils import make_dict_from_file
from local.text_utils import read_mfa_obj

def make_word_phone_align(
    words_info: list, 
    phones_info: list
) -> Dict:
    cur_w_idx = 0  # current word index
    cur_p_idx = 0  # current phone index
    alignment = {}
    while cur_p_idx < len(phones_info) and cur_w_idx < len(words_info):
        cur_w = words_info[cur_w_idx]  # [s, e, word_sym]
        cur_p = phones_info[cur_p_idx]  # [s, e, phone_sym]
        start, end = cur_w[0:2]  # start time stemp / end time stemp for one word
        
        if cur_w_idx not in alignment:
            alignment[cur_w_idx] = {
                'w': cur_w[-1],  # word
                'p': [],  # phones
                's': cur_w[0:2]  # segment info
            }
        
        if (cur_p[0] >= start) and (cur_p[1] <= end):
            alignment[cur_w_idx]['p'].append(cur_p[-1])
            cur_p_idx += 1
        else:
            cur_w_idx += 1
    return alignment


def tokenize(
    alignment: dict, 
    word2id: Optional[Dict] = None, 
    phone2id: Optional[Dict] = None,
    normalize_unk: bool = False, # when phone is unk change the corresponde word as unk
    mode: str = 'train', # if some word not in word2id extend the word2id 
) -> Tuple[Dict, Dict, Dict]:
    tokenized_alignemt = {}
    for idx, info in alignment.items():
        tokenized_alignemt.update({idx: {}})
        word = info['w'] # Chinese CHARACTERS!!! DO NOT SUPPORT Chinese WORD!!
        phones = info['p']

        t_phones = [] # tokenized phone list

        if ('unk' in phones) and normalize_unk:
            word = 'unk'
            phones = ['unk']
        
        if word not in word2id: 
            if (mode == 'test'):
                raise KeyError("{} not in word2id".format(word))
            else:
                max_word_id = len(word2id) # assume word id start from 1 as in most e2e training case 0 is blank
                word2id.update({word: max_word_id + 1})

        for p in phones:
            if p not in phone2id:
                if (mode == 'test'):
                    raise KeyError("{} not in phone2id".format(p))
                else:
                    max_phone_id = len(phone2id)
                    phone2id.update({p: max_phone_id + 1})
            t_phones.append(phone2id[p])
    
        tokenized_alignemt.update({
            idx: {
                'w': word2id[word],
                'p': t_phones,
                's': info['s']
            }
        })
    return (tokenized_alignemt, word2id, phone2id)


class MakeDataList(argparse.ArgumentParser):
    def __init__(self):
        super(MakeDataList, self).__init__()
        self.add_argument(
            '--align_scp', required=True,
            help='MFA alignment scp'
        )
        self.add_argument(
            '--wav_scp', required=False,
            default=None,
            help='wav scp file'
        )
        self.add_argument(
            '--word2id', required=False,
            help='word to id files'
        )
        self.add_argument(
            '--phone2id', required=False,
            help='phone to id file'
        )
        self.add_argument(
            '--mode', required=True,
            default='new',
            help='''extend word id when word in mfa align result not in word2id'''
            '''also will apply on phones'''
        )
        self.add_argument(
            '--dest_path', required=True,
            help='tokenized data list path'
        )
        self.add_argument(
            '--normalize_unk', required=False,
            default=True,
            help='whether map the unk phones correspond words into unk'
        )
        self.add_argument(
            '--utt2spk', required=False,
            default=None,
            help='utterance id to speaker id'
        )
        self.add_argument(
            '--prefix', required=False,
            default='train',
            help='prefix for output files'
        )
        self.parse_args()
        self.init_param()

    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            msg = ('unrecongized arguments %s')
            self.error(msg % " ".join(argv))
        args_dict = args.__dict__
        for key, value in args_dict.items():
            setattr(self, key, value)
    
    def init_param(self):
        if self.align_scp:
            self.align_scp = make_dict_from_file(self.align_scp)
        if self.wav_scp:
            self.wav_scp = make_dict_from_file(self.wav_scp)

        if self.word2id:
            self.word2id = make_dict_from_file(self.word2id)
            self.word2id = {k: int(v) for k, v in self.word2id.items()}
        else:
            self.word2id = {'unk': 1}
        
        if self.phone2id:
            self.phone2id = make_dict_from_file(self.phone2id)
            self.phone2id = {k: int(v) for k, v in self.phone2id.items()}
        else:
            self.phone2id = {'unk': 1}

        if self.utt2spk:
            self.utt2spk = make_dict_from_file(self.utt2spk) 
            self.spk2id = {s:i for i,s in enumerate(self.utt2spk.values())}
            self.utt2spk_id = {u:self.spk2id[s] for u, s in self.utt2spk.items()} 
        
        assert self.mode in ('new', 'test')

        if not os.path.isdir(self.dest_path):
            os.makedirs(self.dest_path)
        
        self.recorder = {
            'raw': open("{}/{}.datalist.raw".format(self.dest_path, self.prefix), 'w'),
            'tokenize': open("{}/{}.datalist".format(self.dest_path, self.prefix), 'w'),
            'word2id':  None if self.mode=='test' else open("{}/word2id".format(self.dest_path), 'w'),
            'phone2id': None if self.mode=='test' else open("{}/phone2id".format(self.dest_path), 'w')
        } 
        

    
    def make(self):
        for utt, ali in self.align_scp.items():
            words_info, phones_info = read_mfa_obj(ali)
            one_align = make_word_phone_align(words_info, phones_info)
            tokenize_align, word2id, phone2id = tokenize(
                one_align, word2id=self.word2id, phone2id=self.phone2id, 
                normalize_unk=self.normalize_unk, append_mode=self.append_mode
            )
            if self.append_mode:
                self.word2id = word2id
                self.phone2id = phone2id
            one_tokenize_obj = {
                'key': utt,
                'bpe_label': [[v['w']] for i,v in tokenize_align.items()],
                'phn_label': [v['p'] for i,v in tokenize_align.items()],
                'segments': [v['s'] for i,v in tokenize_align.items()]
            }
            one_raw_obj = {
                'key': utt,
                'raw_label': [[v['w']] for i,v in one_align.items()],
                'phn_label': [v['p'] for i,v in one_align.items()],
                'segments': [v['s'] for i,v in one_align.items()]
            }
            if self.wav_scp != None:
                if utt not in self.wav_scp:
                    raise KeyError("{} not in wav_scp files".format(utt))
                one_tokenize_obj.update({'sph': self.wav_scp[utt]})
                one_raw_obj.update({'sph': self.wav_scp[utt]})
            
            self.recorder['raw'].write("{}\n".format(json.dumps(one_raw_obj)))
            self.recorder['raw'].flush()
            self.recorder['tokenize'].write("{}\n".format(json.dumps(one_tokenize_obj)))
            self.recorder['tokenize'].flush()

        if self.mode == 'new':
            for word, id in self.word2id.items():
                self.recorder['word2id'].write("{} {}\n".format(word, id))
            for phone, id in self.phone2id.items():
                self.recorder['phone2id'].write("{} {}\n".format(phone, id))
        
        for k, v in self.recorder.items():
            if v != None:
                v.close()
        

if __name__ == '__main__':
    Maker = MakeDataList()
    Maker.make()

