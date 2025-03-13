#!/usr/bin/env python3

import sys
import pandas as pd

#lsd, 1, 0.01, 1.16e-05, 1.1800220585236052e-06, 0.9589974880218506, 0.03703703703703709, 7.50401133547124e-05, 2.7014440807696465

csv = sys.argv[1]
out_csv = sys.argv[2]

data = pd.read_csv(csv, header=None, names=['keyword', 'cost_miss', 'cost_fa', 'prior', 'dcf', 'best_thres', 'p_miss', 'p_fa', 'fa_per_hour'])
#print(data.columns)
#print(data[2])
#print(data[4].mean())
#print(data[8].max())
each_word_data = data[data['keyword'] != 'all']
avg_row = {'keyword': 'avg',
           'cost_miss': int(each_word_data['cost_miss'].mean()),
           'cost_fa': each_word_data['cost_fa'].mean(),
           'prior': f"{(each_word_data['prior'].mean()):.3g}",
           'dcf': each_word_data['dcf'].mean(),
           'best_thres': each_word_data['best_thres'].mean(),
           'p_miss': each_word_data['p_miss'].mean(),
           'p_fa': each_word_data['p_fa'].mean(),
           'fa_per_hour': each_word_data['fa_per_hour'].mean()
           }
data = pd.concat([data, pd.DataFrame([avg_row])], ignore_index=True)
#data = data.append(avg_row, ingore_index=True)
data.to_csv(out_csv, index=False)
