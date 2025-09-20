#!/usr/bin/env python3

import sys
import json

lexicon_path = sys.argv[1]
phone_path = sys.argv[2]
init2fin = sys.argv[3]
fin2init = sys.argv[4]
char_set = sys.argv[5]
tone_group = sys.argv[6]

with open(lexicon_path) as f_lex:
    lexicon_dict = json.load(f_lex)

id2phone_dict = {}
with open(phone_path) as f_phone:
    for line in f_phone:
        phone, pid = line.strip().split()
        id2phone_dict[pid] = phone

init2fin_dict = lexicon_dict["by_init"]
with open(init2fin, 'w') as f_init2fin:
    for init_id in init2fin_dict:
        init = id2phone_dict[init_id]
        fin_list = [ id2phone_dict[str(f)] for f in init2fin_dict[init_id] ]
        f_init2fin.write("{} {}\n".format(init, " ".join(fin_list)))

fin2init_dict = lexicon_dict["by_final"]
with open(fin2init, 'w') as f_fin2init:
    for fin_id in fin2init_dict:
        fin = id2phone_dict[fin_id]
        init_list = [ id2phone_dict[str(i)] for i in fin2init_dict[fin_id] ]
        f_fin2init.write("{} {}\n".format(fin, " ".join(init_list)))

char_list = lexicon_dict["by_len"]["1"]
with open(char_set, 'w') as f_char:
    for char in char_list:
        init_id, fin_id = char[0]
        init = id2phone_dict[str(init_id)]
        fin = id2phone_dict[str(fin_id)]
        f_char.write("[{}, {}], ".format(init, fin))

alt_tone_dict = lexicon_dict["alt_tone"]
alt_tone_list = []
for fin_id in alt_tone_dict:
    fin_alt_list = [ id2phone_dict[str(f)] for f in alt_tone_dict[fin_id] ]
    fin_alt_list.sort()
    if fin_alt_list not in alt_tone_list:
        alt_tone_list.append(fin_alt_list)

with open(tone_group, 'w') as f_tone:
    for fin_alt_list in alt_tone_list:
        f_tone.write("{}\n".format(fin_alt_list))
