import re
import sys
import json
import random


def load_scp(scp_file):
    out_dict = {}
    with open(scp_file, 'r') as f_scp:
        for line in f_scp:
            uttid, content = re.split(r'\s+', line.strip(), maxsplit=1)
            out_dict[uttid] = content
    return out_dict


def load_dict(dict_file):
    my_dict = {}
    with open(dict_file, 'r') as f_dict:
        for line in f_dict:
            key, value = line.strip().split()
            my_dict[key] = int(value)
    return my_dict


def make_datalist(phnid_seq_dict, embedding_scp, out_datalist):
    with open(out_datalist, 'w') as f_datalist:
        n_miss_key = 0
        for key, phnid_seq in phnid_seq_dict.items():
            if key not in embedding_scp:
                n_miss_key += 1
                continue
            embedding_path = embedding_scp[key]
            # phn_label = [ phone2id[phn] for phn in phn_seq ]
            one_obj = {
                'key': key,
                'phn_label': phnid_seq,
                'sph_emb': embedding_path
            }
            f_datalist.write(f"{json.dumps(one_obj)}\n")
            # f_datalist.write(f"{key} {phn_seq} {embedding_path}\n")
    print(f"num of mismatch key: {n_miss_key}")


def load_timit_lexicon(lexicon):
    word2phn = {}
    with open(lexicon, 'r') as f_lexicon:
        for line in f_lexicon:
            if line.strip().startswith(';'):
                continue
            word, phones = re.split(r'\s+', line.strip(), maxsplit=1)
            phn_seq = phones.strip('/').split()
            if word.endswith('~n') or word.endswith('~v'):
                raw_word, type = word.split('~')
                if raw_word not in word2phn:
                    word2phn[raw_word] = {}
                word2phn[raw_word][type] = phn_seq
            else:
                word2phn[word] = phn_seq
    print(f"len of word2phn: {len(word2phn)}")
    return word2phn


def load_list(list_file):
    with open(list_file, 'r') as f_list:
        return [line.strip() for line in f_list]


def make_phone2id(phones, out_phone2id, unk='unk', unk_id=0, start_id=3):
    phone2id = {}
    for i, phn in enumerate(phones):
        phone2id[phn] = i + start_id
    phone2id[unk] = unk_id
    with open(out_phone2id, 'w') as f_phone2id:
        for phn, id in phone2id.items():
            f_phone2id.write(f"{phn} {id}\n")
    return phone2id


def main(text_scp, lexicon, phones, embedding_scp, out_phone2id, out_datalist):
    unk = 'unk'
    seed = 42
    random.seed(seed)
    text_dict = load_scp(text_scp)
    word2phn = load_timit_lexicon(lexicon)
    phones = load_list(phones)
    phone2id = make_phone2id(phones, out_phone2id, unk='unk', unk_id=0, start_id=3)

    phn_seq_dict = {}
    oov_words = set()
    n_invalid_utt = 0
    for key, text in text_dict.items():
        phn_seq = []
        invalid_utt = False
        for word in text.split():
            word = re.sub('[0-9\.,?!:;"]', '', word)
            word = word.lower()
            if word in word2phn:
                word_phn = word2phn[word]
                if type(word_phn) == dict:
                    word_phn = random.choice(list(word_phn.values()))
                    phn_seq.extend(word_phn)
                else:
                    phn_seq.extend(word_phn)
            else:
                oov_words.add(word)
                invalid_utt = True
                break
        if invalid_utt:
            n_invalid_utt += 1
            continue
        phn_seq_dict[key] = phn_seq
    print(f"num of oov words: {len(oov_words)}")
    print(f"first 10 oov words: {list(oov_words)[:10]}")
    print("skipped {} utts for oov word".format(n_invalid_utt))

    phnid_seq_dict = {}
    oov_phones = set()
    for key, phn_seq in phn_seq_dict.items():
        phnid_seq = []
        for phn in phn_seq:
            phn = re.sub(r'[0-9]', '', phn)
            if phn in phones:
                phnid_seq.append(phone2id[phn])
            else:
                phnid_seq.append(phone2id[unk])
                oov_phones.add(phn)
        phnid_seq_dict[key] = phnid_seq
    print(f"num of oov phones: {len(oov_phones)}")
    print(f"first 10 oov phones: {list(oov_phones)[:10]}")

    embedding_scp = load_scp(embedding_scp)

    make_datalist(phnid_seq_dict, embedding_scp, out_datalist)


if __name__ == '__main__':
    if len(sys.argv) != 7:
        print (f"Usage: python {sys.argv[0]} text_scp lexicon phones embedding_scp out_phone2id out_datalist")
        exit()
    text_scp = sys.argv[1]
    lexicon = sys.argv[2]
    phones = sys.argv[3]
    embedding_scp = sys.argv[4]
    out_phone2id = sys.argv[5]

    out_datalist = sys.argv[6]
    main(text_scp, lexicon, phones, embedding_scp, out_phone2id, out_datalist)
