from pyhanlp import *
import re
import sys
import json
import jieba
import time

word_pinyin_cache = {}
pattern = re.compile(r"([bpmfdtnlgkhjqxzrzcsyw]?h?)([aeiouüv]+[nr]?g?[\d]?)")

def convert_and_split(segment):
    word2phone = {}
    for i, word in enumerate(segment):
        word2phone[i] = []
        if word.word not in word_pinyin_cache:
            phone_seq = HanLP.convertToPinyinList(word.word)
            phone_seq = str(phone_seq).strip("[]").replace(" ","").split(",")
            word_pinyin_cache[word.word] = phone_seq
        else:
            phone_seq = word_pinyin_cache[word.word]
        if len(word.word) != len(phone_seq):
            continue
        for j, character in enumerate(phone_seq):
            if character == 'none5':
                word2phone[i].append([word.word[j], 'UNK', 'UNK'])
            else:
                match = pattern.match(character.replace(" ", ""))
                if match:
                    init, final = match.groups()
                    if init == "":
                        init = None
                    word2phone[i].append([word.word[j], init, final])
                else:
                    word2phone[i].append([word.word[j], None])
    return word2phone

def read_scp(scp_file):
    #print("read_scp(): scp_file is {}".format(scp_file))
    obj = {}
    with open(scp_file) as ff:
        for line in ff.readlines():
            line = line.strip()
            line = re.sub(r"\s+", " ", line)
            if '\t' in line:
                line = line.replace("\t", " ")
            line = line.split(" ")
            if len(line) < 2:
                continue
            utt = line[0]
            content = "".join(line[1:])
            obj.update({utt: content})
    return obj

def detach_item(items, word2id, phone2id):
    max_wrd_id = max(list(word2id.values())) if len(word2id) > 0 else 2
    max_phn_id = max(list(phone2id.values())) if len(phone2id) > 0 else 2
    word_seq = []
    phn_seq = []
    for item in items:
        word, init, final = item
        if word not in word2id:
            max_wrd_id += 1
            word2id[word] = max_wrd_id
        if init != None and init not in phone2id:
            max_phn_id += 1
            phone2id[init] = max_phn_id
        if final not in phone2id:
            max_phn_id += 1
            phone2id[final] = max_phn_id
        word_seq.append(word2id[word])
        if init != None:
            phn_seq.append([phone2id[init], phone2id[final]])
        else:
            phn_seq.append([phone2id[final]])

    return word_seq, phn_seq, word2id, phone2id

def multi_index(lexicon):
    lexicon_by_len = lexicon
    lexicon_by_init = {}
    lexicon_by_final = {}
    lexicon_multi_index = {}
    #for i in lexicon_by_len:
    #    print(i)
    for char in lexicon_by_len[1]:
        char = char[0]
        if len(char) != 2:
            #print("char: {}".format(char))
            continue
        init, final = char
        if init not in lexicon_by_init:
            lexicon_by_init[init] = []
        if final not in lexicon_by_init:
            lexicon_by_init[init].append(final)
        if final not in lexicon_by_final:
            lexicon_by_final[final] = []
        if init not in lexicon_by_final:
            lexicon_by_final[final].append(init)
    lexicon_multi_index['by_len'] = lexicon_by_len
    lexicon_multi_index['by_init'] = lexicon_by_init
    lexicon_multi_index['by_final'] = lexicon_by_final
    return lexicon_multi_index

def split_and_tokenize(src_data, src_id2word, in_word2id, in_phone2id):
    #print("split_and_tokenize() start")
    word2id = in_word2id
    phone2id = in_phone2id
    #tokenized_objs = {}
    lexicon = {}
    out_data = []
    max_wordid = get_max_value(in_word2id)
    max_phoneid = get_max_value(in_phone2id)
    t_cost_till_seg = 0
    t_cost_till_convert = 0
    t_cost_till_items = 0
    t_cost_till_detach = 0
    t_cost_till_lex = 0
    for i, utt_json in enumerate(src_data):
    #for i, (key, content) in enumerate(text_obj.items()):
        #print(i)
        t_start_seg = time.time()
        utt_dict = json.loads(utt_json)
        #key = utt_dict['key']
        bpe_label = utt_dict['bpe_label']
        finetune_data = utt_dict['finetune_data']
        if finetune_data == 'target_neg':
            len_bpe_label = len(utt_dict['bpe_label'])
            utt_dict['bpe_label'] = [max_wordid+1] * len_bpe_label
            utt_dict['phn_label'] = [ [max_phoneid+1, max_phoneid+1] for i in range(len_bpe_label) ]
            out_data.append(utt_dict)
            continue
        content = ""
        for bpe in bpe_label:
            content += src_id2word[bpe]

        match_digit_letter = re.search(r'[a-zA-Z0-9]', content) # remove the content which contain digit and letter 
                                                                # digit in chinese format is allowed for example "一二三"
        if match_digit_letter:
            print("match digit or letter: {}".format(content))
            continue
        if i % 1000 == 0:
            print ("hanlp has segment {} utterance".format(i))
            #print("time cost till seg: {:.3f}".format(t_cost_till_seg))
            #print("time cost till convert: {:.3f}".format(t_cost_till_convert))
            #print("time cost till items: {:.3f}".format(t_cost_till_items))
            #print("time cost till detach: {:.3f}".format(t_cost_till_detach))
            #print("time cost till lex: {:.3f}".format(t_cost_till_lex))
        one_sample_wrd = []
        one_sample_phn = []
        one_sample_phn_seg = []
        segment = HanLP.segment(content)
        t_cost_till_seg += time.time() - t_start_seg
        t_start_convert = time.time()
        #print("content: {}".format(content))
        #print("segment: {}".format(segment))
        #segment_label = [ [ cid for cid, character in enumerate(str(seg).split('/')[0]) ] for seg in segment ]
        word2phone = convert_and_split(segment)
        t_cost_till_convert += time.time() - t_start_convert
        #segment_label = [ [ [ p for p in py[1:3] if p != None ] for py in items ] for idx, items in word2phone.items() ]
        #segment_label = [ [ p for p in py[1:3] if p != None for py in items ] for idx, items in word2phone.items() ]
        #segment_label = [ [ py for py in items[1:3] if py != None ] for idx, items in word2phone.items() ]
        #print("segment_label: {}".format(segment_label))
        t_start_items = time.time()
        for idx, items in word2phone.items():
            t_start_detach = time.time()
            word_seq, phn_seq, word2id, phone2id = detach_item(items, word2id, phone2id)
            t_cost_till_detach += time.time() - t_start_detach
            one_sample_wrd.extend(word_seq)
            one_sample_phn.extend(phn_seq)
            one_sample_phn_seg.append(phn_seq)
            seg_len = len(phn_seq)
            #if seg_len == 0:
            #    print(phn_seq)
            t_start_lex = time.time()
            if seg_len not in lexicon:
                lexicon[seg_len] = set()
                #lexicon[seg_len] = []
            
            lexicon[seg_len].add(json.dumps({'phn_seq': phn_seq}))
            t_cost_till_lex += time.time() - t_start_lex
            #print("word_seq: {}".format(word_seq))
            #print("phn_seq: {}".format(phn_seq))
        utt_dict['bpe_label'] = one_sample_wrd
        utt_dict['phn_label'] = one_sample_phn
        utt_dict['segment_label'] = one_sample_phn_seg
        out_data.append(utt_dict)
        #tokenized_objs.update({
        #    key: {'bpe_label': one_sample_wrd, 'phn_label': one_sample_phn, 'segment_label': one_sample_phn_seg}
        #})
        t_cost_till_items += time.time() - t_start_items
    #for seg_len in lexicon:
    #    print(seg_len)
    for seg_len, phn_seqs in lexicon.items():
        #print(seg_len, phn_seqs)
        phn_seqs = list(phn_seqs)
        phn_seqs = [ json.loads(p)['phn_seq'] for p in phn_seqs ]
        lexicon[seg_len] = phn_seqs
    lexicon = multi_index(lexicon)
    print("length of lexicon: {}".format(len(lexicon)))

    return out_data, word2id, phone2id, lexicon

def write_data_list(out_data, word2id, phone2id, out_word2id, out_phone2id, out_datalist):
    dlist = open(out_datalist, 'w')
    for data in out_data:
        dlist.write(f"{json.dumps(data)}\n")
    #for key, tokenized_item in tokenized_objs.items():
    #    if key not in wav_scp:
    #        continue
    #    word_seq = tokenized_item['bpe_label']
    #    phn_seq = tokenized_item['phn_label']
    #    segment_label = tokenized_item['segment_label']
    #    one_obj = {
    #        'key': key,
    #        'sph': wav_scp[key],
    #        'bpe_label': word_seq,
    #        'phn_label': phn_seq,
    #        'segment_label': segment_label,
    #        'kw_candidate': [x for x in range(len(phn_seq))],
    #        'b_kw_candidate': [x for x in range(len(word_seq))]
    #    }
    #    dlist.write(f"{json.dumps(one_obj)}\n")
    
    with open(out_word2id, 'w') as wf:
        for word, id in word2id.items():
            wf.write("{} {}\n".format(word, id))
    
    with open(out_phone2id, 'w') as pf:
        for phn, id in phone2id.items():
            pf.write("{} {}\n".format(phn, id))

def load_dict(dict_file):
    my_dict = {}
    with open(dict_file, 'r') as f_dict:
        for line in f_dict:
            key, value = line.strip().split()
            my_dict[key] = int(value)
    return my_dict

def load_dict_reverse(dict_file):
    my_dict = {}
    with open(dict_file, 'r') as f_dict:
        for line in f_dict:
            key, value = line.strip().split()
            my_dict[int(value)] = key
    return my_dict

def write_lexicon(lexicon, out_lexicon):
    print(len(lexicon))
    with open(out_lexicon, 'w') as f_lexicon:
        json.dump(lexicon, f_lexicon, indent=4)

def read_datalist(datalist):
    data = []
    with open(datalist) as f:
        for line in f:
            data.append(line.strip())
    return data

def get_max_value(dict):
    max_value = 0
    for key, value in dict.items():
        if value > max_value:
            max_value = value
    return max_value

def main(src_datalist, src_word2id, in_word2id, in_phone2id, out_word2id, out_phone2id, out_datalist, out_lexicon):
    src_data = read_datalist(src_datalist)
    src_id2word_dict = load_dict_reverse(src_word2id)
    #text_scp = read_scp(text_scp)
    in_word2id = load_dict(in_word2id)
    in_phone2id = load_dict(in_phone2id)
    #print("len of text_scp: {}".format(len(text_scp)))
    out_data, word2id, phone2id, lexicon = split_and_tokenize(src_data, src_id2word_dict, in_word2id, in_phone2id)
    #write_lexicon(lexicon['by_len'], out_lexicon)
    write_lexicon(lexicon, out_lexicon)
    #wav_scp = read_scp(wav_scp)
    write_data_list(out_data, word2id, phone2id, out_word2id, out_phone2id, out_datalist)

if __name__ == '__main__':
    if len(sys.argv) != 9:
        print ("Usage: python hanlp_split.py src_datalist src_word2id in_word2id in_phone2id out_word2id out_phone2id out_datalist out_lexicon")
        exit()
    src_datalist = sys.argv[1]
    src_word2id = sys.argv[2]
    #text_scp = sys.argv[1]
    #wav_scp = sys.argv[2]
    in_word2id = sys.argv[3]
    in_phone2id = sys.argv[4]
    out_word2id = sys.argv[5]
    out_phone2id = sys.argv[6]
    out_datalist = sys.argv[7]
    out_lexicon = sys.argv[8]
    main(src_datalist, src_word2id, in_word2id, in_phone2id, out_word2id, out_phone2id, out_datalist, out_lexicon)
