import json
import os
import copy
import torchaudio
import random
from argparse import ArgumentParser
from utils import make_dict_from_file

def read_datalist(datalist_file):
    datalist = []
    with open(datalist_file) as df:
        for line in df.readlines():
            line = line.strip()
            datalist.append(json.loads(line))
    return datalist

def combine_sym2id(new_sym2id, org_sym2id):
        max_new_ids = max(list(new_sym2id.values()))
        org_keys = list(org_sym2id.keys())
        num_org = 1
        for k in org_keys:
            if k in new_sym2id:
                continue
            else:
                new_sym2id.update({k:max_new_ids+num_org})
                num_org += 1
        return new_sym2id

class MakeDataList(ArgumentParser):
    def __init__(self):
        super(MakeDataList, self).__init__()
        self.add_argument("--wav_scp", required=False, default=None)
        self.add_argument("--trans", required=False, default=None)
        self.add_argument("--spk2utt", required=False, default=None)
        self.add_argument("--utt2spk", required=False, default=None)
        self.add_argument("--utt2seg", required=False, default=None)
        self.add_argument("--utt2keyword", required=False, default=None)
        self.add_argument("--subsample_class", required=False, default=None)
        self.add_argument("--utts_per_class", required=False, default=None)
        self.add_argument("--word_list", required=False, default=None)
        self.add_argument("--max_word_frequency", required=False, default=2500)
        self.add_argument("--win_shift", required=False, type=int, default=0)
        self.add_argument("--win_len", required=False, type=int, default=0)
        self.add_argument("--exist_datalist", required=False, default=None)
        self.add_argument("--align", required=False, default=None)
        self.add_argument(
            "--keyword2id", required=False, default=None,
            help="""Generate new datalist does not required keyword2id, this file is required when 
            append new data into exist datalist"""
        )
        self.add_argument(
            "--spk2id", required=False, default=None,
            help="""Generate new datalist does not required spk2id, this file is required when 
            append new data into exist datalist"""
        )
        self.add_argument(
            "--proc", required=True,  default='New', 
            help="""[New | Append | Subsample] New means create a new datalist; Append means add new 
            data into existing datalist; Subsample means subsample datalist from existing datalist"""
        )
        self.add_argument(
            "--dest_dir", required=True,
            help="output dir"     
        )
        self.add_argument(
            "--prefix", required=False,
            default='',
            help="""prefix for output files e.g. datalist.txt when specific --prefix it will be save as
            prefix.datalist
            """
        )
        self.parse_args()
        self.init_param()
    # DON't modify this function
    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            msg = ('unrecognized arguments: %s')
            self.error(msg % ' '.join(argv))
        args_dict = args.__dict__
        for key, value in args_dict.items():
            setattr(self, key, value)
        
    # read files 
    def init_param(self):
        if self.wav_scp:
            self.wav_dict = make_dict_from_file(self.wav_scp)

        if self.trans:
            self.trans = make_dict_from_file(self.trans)

        if self.spk2utt:
            self.spk2utt = make_dict_from_file(self.spk2utt)

        if self.utt2spk:
            self.utt2spk = make_dict_from_file(self.utt2spk)

        if self.utt2keyword:
            self.utt2keyword = make_dict_from_file(self.utt2keyword)
            for i, key in enumerate(self.utt2keyword.keys()):
                value = self.utt2keyword[key]
                if len(value.split(" ")) != 1:
                    try:
                        keyword, head, tail = value.split(" ")
                        self.utt2keyword[key] = [keyword, float(head), float(tail)]
                    except:
                        print ("""--utt2keyword file should formatted as utt_id: keyword or """
                         """utt_id: keyword keyword_start_point keyword endpoint, format error in line """
                        """{} utt_id: {} {}\n""".format(i+1, key, value))
                        exit()
                else:
                    continue

        if self.align:
            self.align = make_dict_from_file(self.align)
            self.read_mfa_align()
        else:
            self.utt2align = {}

        if self.exist_datalist:
            self.exist_datalist = read_datalist(self.exist_datalist)
        
        if self.keyword2id:
            self.keyword2id = make_dict_from_file(self.keyword2id)
            self.keyword2id = {k:int(v) for k,v in self.keyword2id.items()}
            self.id2keyword = {v:k for k,v in self.keyword2id.items()}
        
        if self.spk2id:
            self.spk2id = make_dict_from_file(self.spk2id)
            self.spk2id = {k: int(v) for k, v in self.spk2id.items()}

        if self.subsample_class:
            self.subsample_class = int(self.subsample_class)

        if self.utts_per_class:
            self.utts_per_class = int(self.utts_per_class)

        if self.win_shift != 0:
            self.streaming = True
            if self.win_len == 0:
                raise AssertionError("win shift has been specified, but miss win_len!!!") 
            else:
                print(
                    """win_shift has been specified, segment will be setting by
                    win_shift: {} and win_len:{} """.format(self.win_shift, self.win_len)
                )
        else:
            self.streaming = False
        
        if not os.path.isdir(self.dest_dir):
            os.makedirs(self.dest_dir)

    def append_keyword_datalist(self, datalist, new_keyword2id, new_spk2id=None):
        if self.exist_datalist == None:
            return datalist
        
        assert (self.keyword2id != None)
        new_keyword2id = combine_sym2id(new_keyword2id, self.keyword2id)

        if new_spk2id != None:
            assert (self.spk2id != None)
            new_spk2id = combine_sym2id(new_spk2id, self.spk2id)
        else:
            new_spk2id = None

        for d in self.exist_datalist:
            org_id = d['word_keyword'][0]
            org_keyword = self.id2keyword[org_id]
            d.update({'word_keyword': [new_keyword2id[org_keyword]]})
            if new_spk2id != None:
                org_spk_id = d['speaker'][0]
                org_spk = self.spk2id[org_spk_id]
                d.update({'speaker': [new_spk2id[org_spk]]})
        
        combined_list = datalist + self.exist_datalist 
        return combined_list, new_keyword2id, new_spk2id

    def combine_sym2id(new_sym2id, org_sym2id):
        max_new_ids = max(list(new_sym2id))
        org_keys = list(org_sym2id.keys())
        num_org = 1
        for k in org_keys:
            if k in new_sym2id:
                continue
            else:
                new_sym2id.update({k:max_new_ids+num_org})
                num_org += 1
        return new_sym2id

    def read_mfa_align(self):
        self.utt2align = {}
        for i, (utt, align_json) in enumerate(self.align.items()):
            one_align = json.loads(open(align_json).readline())
            word_align_info = one_align['tiers']['words']['entries']
            phone_align_info = one_align['tiers']['phones']['entries']
            self.utt2align.update({
                utt:{'word_align_info': word_align_info, 'phone_align_info': phone_align_info}
            })

    def make_keyword_datalist(self, keyword2id=None):
        datalist = []
        if keyword2id != None:
            user_keyword2id = True
            print ("""keyword2id has been provided, and all processes will be executed according """
                   """to the standards in File keyword2id. Please check if it is correct""")
        else:
            user_keyword2id = False
            keyword2id = {}
            kw_id = 0

        if self.utt2spk:
            spks = list(self.utt2spk.values())
            spks = list(set(spks))
            spk2id = {s: i for i, s in enumerate(spks)}
        else:
            spk2id = None

        assert self.utt2keyword != None
        for utt, wav_file in self.wav_dict.items():
            one_segments = []
            keyword = self.utt2keyword[utt]
            if isinstance(keyword, list):
                keyword, start, end = keyword
                segment = [start, end]
            else:
                segment = None
            
            if (user_keyword2id) and (keyword not in keyword2id) :
                raise AssertionError(
                    "keyword {} of utt {} not in --keyword2id files".format(keyword, utt)
                )
            if keyword not in keyword2id:
                keyword2id[keyword] = kw_id
                keyword_id = kw_id
                kw_id += 1
            else:
                keyword_id = keyword2id[keyword]

            if utt in self.utt2align:
                for item in self.utt2align[utt]['word_align_info']:
                    start, end, word = item
                    if word == keyword:
                        segment = [start, end]
                        one_segments.append(segment)
            elif segment != None:
                one_segments.append(segment)
            elif self.streaming:
                one_segments  = self.make_streaming(wav_file)
            else:
                one_segments.append([0.01, 0.01])

            for i, seg in enumerate(one_segments):
                one_obj = {}
                one_obj.update({
                    'key': "{}_{}".format(utt, i),
                    'sph': wav_file,
                    'segment': copy.deepcopy(seg),
                    'word_keyword': [keyword_id]
                })
                if self.utt2spk:
                    spk_id = spk2id[self.utt2spk[utt]] 
                    one_obj.update({
                        'speaker': [spk_id]
                    })
                datalist.append(one_obj)
            
        return datalist, keyword2id, spk2id

    def make_streaming(self, wav_file):
        wav, sr = torchaudio.load(wav_file)
        segments = []
        if sr != 16000:
            print ("Sample rate of {} is not 16k".format(wav_file))
        num_samples = wav.size(1)
        num_segment = (num_samples - self.win_len) // self.win_shift - 1
        num_segment = 1 if num_samples <= 0 else num_segment
        for x in range(num_segment):
            start = x * self.win_shift
            end = x * self.win_shift + self.win_len
            segments.append([start, end])
        return segments

    def subsample_kw_datalist(self):
        assert self.keyword2id != None
        id2keyword = {v:k for k,v in self.keyword2id.items()}
        keyword2list = {} 
        for d in self.exist_datalist:
            kw_id = d['word_keyword'][0]
            if kw_id not in keyword2list:
                keyword2list[kw_id] = [d]
            else:
                keyword2list[kw_id].append(d)

        if self.utts_per_class:
            for kw_id in list(keyword2list.keys()):
                if len(keyword2list[kw_id]) < self.utts_per_class:
                    keyword2list.pop(kw_id)
        
        candidate_kw_ids = list(keyword2list.keys())
        random.shuffle(candidate_kw_ids)
        target_kw_ids = candidate_kw_ids[0: self.subsample_class]

        datalist = []
        keyword2id = {}
        for kw_id in target_kw_ids:
            one_datalist = keyword2list[kw_id]
            random.shuffle(one_datalist)
            target_datalist = one_datalist[0:self.utts_per_class if self.utts_per_class else len(datalist)]
            datalist.extend(target_datalist)
            keyword2id.update({id2keyword[kw_id]: kw_id})
        #TODO: add spk2id 
        print (keyword2id)
        return datalist, keyword2id, None


    def record_result(self, datalist, keyword2id=None, spk2id=None):
        new_datalist_file = open("{}/{}datalist.txt".format(self.dest_dir, self.prefix), "w")
        for d in datalist:
            one_obj = json.dumps(d)
            new_datalist_file.write(one_obj + "\n")
        new_datalist_file.close()
        if keyword2id != None:
            new_keyword2id_file = open("{}/{}keyword2id".format(self.dest_dir, self.prefix), "w")    
            for keyword, id in keyword2id.items():
                new_keyword2id_file.write("{} {}\n".format(keyword, id))
            new_keyword2id_file.close() 
        if spk2id != None:
            new_spk2id_file = open("{}/{}spk2id".format(self.dest_dir, self.prefix), "w")    
            for spk, id in spk2id:
                new_spk2id_file.write("{} {}\n".format(spk, id))
            new_spk2id_file.close()

def main():
    maker = MakeDataList()
    if maker.proc == 'Append':
        new_datalist, new_keyword2id, new_spk2id = maker.make_keyword_datalist()
        datalist, keyword2id, spk2id = maker.append_keyword_datalist(
            new_datalist, new_keyword2id, new_spk2id
        )
        print ("Append new datalist into exist datalist")
    elif maker.proc == 'Subsample':
        datalist, keyword2id, spk2id = maker.subsample_kw_datalist()
        print ("Subsample datalist")
    elif maker.proc == 'New':
        keyword2id = maker.keyword2id
        datalist, keyword2id, spk2id = maker.make_keyword_datalist(keyword2id=keyword2id)
        print ("Create New datalist")
    else:
        raise NotImplementedError("only support Append Subsample now")
    
    maker.record_result(
        datalist=datalist, keyword2id=keyword2id, spk2id=spk2id
    )
    # TO BE CONTINUED
if __name__ == '__main__':
    main()