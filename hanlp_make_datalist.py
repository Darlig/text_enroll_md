from pyhanlp import *
import re
import sys
import json

def convert_and_split(segment):
    word2phone = {}
    for i, word in enumerate(segment):
        word2phone[i] = []
        phone_seq = HanLP.convertToPinyinList(word.word)
        phone_seq = str(phone_seq).strip("[]").replace(" ","").split(",")
        if len(word.word) != len(phone_seq):
            continue
        for j, character in enumerate(phone_seq):
            if character == 'none5':
                word2phone[i].append([word.word[j], 'UNK', 'UNK'])
            else:
                match = re.match(r"([bpmfdtnlgkhjqxzrzcsyw]?h?)([aeiouüv]+[nr]?g?[\d]?)", character.replace(" ", ""))
                if match:
                    init, final = match.groups()
                    word2phone[i].append([word.word[j], init, final])
                else:
                    word2phone[i].append([word.word[j], None])
    return word2phone



def read_scp(scp_file):
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
        if init not in phone2id:
            max_phn_id += 1
            phone2id[init] = max_phn_id
        if final not in phone2id:
            max_phn_id += 1
            phone2id[final] = max_phn_id
        word_seq.append(word2id[word])
        phn_seq.append([phone2id[init], phone2id[final]])
    return word_seq, phn_seq, word2id, phone2id

def split_and_tokenize(text_obj):
    word2id = {}
    phone2id = {}
    tokenized_objs = {}
    for i, (key, content) in enumerate(text_obj.items()):
        match_digit_letter = re.search(r'[a-zA-Z0-9]', content) # remove the content which contain digit and letter 
                                                                # digit in chinese format is allowed for example "一二三"
        if match_digit_letter:
            continue
        if i % 1000 == 0:
            print ("hanlp has segment {} utterance".format(i))
        one_sample_wrd = []
        one_sample_phn = []
        segment = HanLP.segment(content)
        word2phone = convert_and_split(segment)
        for idx, items in word2phone.items():
            word_seq, phn_seq, word2id, phone2id = detach_item(items, word2id, phone2id)
            one_sample_wrd.extend(word_seq)
            one_sample_phn.extend(phn_seq)
        tokenized_objs.update({
            key: {'bpe_label': one_sample_wrd, 'phn_label': one_sample_phn}
        })
    return tokenized_objs, word2id, phone2id

def write_data_list(wav_scp, tokenized_objs, word2id, phone2id):
    dlist = open('datalist.txt', 'w')
    for key, tokenized_item in tokenized_objs.items():
        if key not in wav_scp:
            continue
        word_seq = tokenized_item['bpe_label']
        phn_seq = tokenized_item['phn_label']
        one_obj = {
            'key': key,
            'sph': wav_scp[key],
            'bpe_label': word_seq,
            'phn_label': phn_seq,
            'kw_candidate': [x for x in range(len(phn_seq))],
            'b_kw_candidate': [x for x in range(len(word_seq))]
        }
        dlist.write(f"{json.dumps(one_obj)}\n")
    
    with open('word2id.txt', 'w') as wf:
        for word, id in word2id.items():
            wf.write("{} {}\n".format(word, id))
    
    with open('phone2id.txt', 'w') as pf:
        for phn, id in phone2id.items():
            pf.write("{} {}\n".format(phn, id))

def main(text_scp, wav_scp):
    text_scp = read_scp(text_scp)
    tokenized_objs, word2id, phone2id = split_and_tokenize(text_scp)
    wav_scp = read_scp(wav_scp)
    write_data_list(wav_scp, tokenized_objs, word2id, phone2id)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ("Usage: python hanlp_split.py text wav.scp")
        exit()
    text_scp = sys.argv[1]
    wav_scp = sys.argv[2]
    main(text_scp, wav_scp)
