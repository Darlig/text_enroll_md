import torch
import yaml
import sys
import os
import argparse

def compute_avg():
    pass

def compute_top():
    pass
exp_dir = sys.argv[1] +"/"
exp_config = exp_dir + 'exp.yaml'
exp_config = yaml.load(open(exp_config), Loader=yaml.FullLoader)
min_epoch = int(sys.argv[2])
max_epoch = int(sys.argv[3])
exp_name = exp_config['exp_name']

valid_ckpt = {k:0 for k in range(min_epoch, max_epoch)}
valid_loss = []

for e in range(min_epoch, max_epoch):
    ckpt = "{}/{}_{}.pt".format(exp_dir, exp_name, e)
    if not os.path.isfile(ckpt):
        print ("No such model {}".format(ckpt))
        exit()
    ckpt = torch.load(ckpt, map_location='cpu')
    one_valid_loss = ckpt['cv_loss']['total_loss'].item()
    valid_ckpt[e] = ckpt['model']
    valid_loss.append(one_valid_loss)
l = list(range(len(valid_loss)))
idx = sorted(l, key=lambda k: valid_loss[k])
min_idx = idx[:max_epoch-min_epoch] 

state = None
avg = None
for k in min_idx:
    k = k + min_epoch
    state = valid_ckpt[k]
    if avg == None:
        avg = state
    else:
        for k in avg.keys():
            avg[k] += state[k]
dst_path = "{}/{}_avg.pt".format(exp_dir, exp_name)
for k in avg.keys():
    if avg[k] is not None:
        avg[k] = torch.true_divide(avg[k], len(min_idx))
torch.save({'model':avg}, dst_path)
