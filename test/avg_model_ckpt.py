import argparse
import os
import copy
import yaml

import torch
import torch.nn
import torch.distributed as dist
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from yamlinclude import YamlIncludeConstructor
from data.loader.data_loader import Dataset
from local.utils import WarmUpLR, read_list, Recorder
from torch.utils.tensorboard import SummaryWriter
import re

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    '--config',
    #    required=True,
    #    help="config file in yaml format e.g. config/ref.yaml"
    #)

    parser.add_argument(
        '--ckpt',
        required=True,
        help="checkpoint file name example e.g. exp/kw_att_5.pt"
    )
    parser.add_argument(
        '--min_epoch',
        required=True,
        type=int,
        help = "min epoch for avgerage"
    )
    parser.add_argument(
        '--max_epoch',
        required=True,
        type=int,
        help = "max epoch for avgerage"
    )

    args = parser.parse_args()
    return args


class Trainer():
    def __init__(
        self,
        #config_file: dict,
    ):
        # init config info
        #self.config_file = config_file
        #self.data_config = config_file['data_config']
        #self.exp_config = config_file['exp_config']
        ckpt_path = args.ckpt
        self.ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_name = os.path.basename(ckpt_path)
        self.ckpt_prefix = re.sub("_\d*.pt$", "", ckpt_name)
        #self.exp_config.update(exp_dir=self.ckpt_dir)
        #self.recorder = Recorder(self.exp_config)


    def avg_model_custom(self):
        min_epoch = args.min_epoch
        max_epoch = args.max_epoch
        avg_epoch = max_epoch - min_epoch
        valid_ckpt = {k:0 for k in range(min_epoch, max_epoch)}
        valid_loss = []
        for e in range(min_epoch, max_epoch):
            ckpt = "{}/{}_{}.pt".format(self.ckpt_dir,self.ckpt_prefix,e)
            ckpt = torch.load(ckpt, map_location='cpu')
            one_valid_loss = ckpt['cv_loss']['total_loss'].item()
            valid_ckpt[e] = ckpt['model']
            valid_loss.append(one_valid_loss)
        sort_idx = sorted(range(len(valid_loss)), key=lambda k: valid_loss[k])
        min_idx = sort_idx[:avg_epoch]
        state = None
        avg_model = None
        for k in min_idx:
            k = k + min_epoch
            state = valid_ckpt[k]
            if avg_model == None:
                avg_model = state
            else:
                for k in avg_model.keys():
                    avg_model[k] += state[k]
        for k in avg_model.keys():
            if avg_model[k] is not None:
                avg_model[k] = torch.true_divide(avg_model[k], avg_epoch)
        torch.save({'model': avg_model}, f"{self.ckpt_dir}/{self.ckpt_prefix}_avg_{min_epoch}-{max_epoch}.pt")
        #self.recorder.save_state(avg_model, epoch='avg_{}-{}'.format(min_epoch, max_epoch))
        
    def run(self):
        # avg model step
        self.avg_model_custom()
        

if __name__ == '__main__':
    args = get_args()
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    #config = yaml.load(open(args.config),Loader=yaml.FullLoader)
    trainer = Trainer()
    #trainer = Trainer(config)
    trainer.run()
