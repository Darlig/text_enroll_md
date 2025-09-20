#!/usr/bin/env python3

import sys
import re

source_text_scp = sys.argv[1]
clean_text_scp = sys.argv[2]

n_err_line = 0
with open(source_text_scp) as f_text, open(clean_text_scp, 'w') as f_clean:
    for line in f_text:
        line_sp = re.split(r'\s+', line.strip(), maxsplit=1)
        if len(line_sp) != 2:
            n_err_line += 1
            continue
        uttid, text = line_sp
        clean_text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        if clean_text == '':
            n_err_line += 1
            continue
        f_clean.write('{} {}\n'.format(uttid, clean_text))
print("Finish with skiping {} error lines".format(n_err_line))
