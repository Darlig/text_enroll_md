import os
import argparse
import copy
import yaml

import torch
import torch.nn as nn
import soundfile as sf

from torch.utils.data import DataLoader
from data.loader.data_loader import Dataset
from local.utils import (
    make_dict_from_file, read_list, Recorder, 
    vinterplate, compute_eer_skleanr, compute_topk
)
from yamlinclude import YamlIncludeConstructor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        required=False,
        help="config file in yaml format e.g. config/ref.yaml"
    )
    parser.add_argument(
        '--keyword2id',
        required=False,
        default=None,
        help="keyword to id files"
    )
    parser.add_argument(
        '--device',
        default='0,1',
        help=' \'0,1\', ==> means use gpu_0 and gpu_1; \'cpu\' means use cpu'
    )
    parser.add_argument(
        '--batch_size',
        default=512,
        type=int,
        help=' test batch size'
    )
    parser.add_argument(
        '--test_ckpt',
        default='avg',
        help='test num ckpt e.g. 30: means test the 30 th epoch model avg means the final model'
    )
    parser.add_argument(
        '--acc_data_list',
        required=False,
        default=None,
        help="data list used to test accuracy"
    )
    parser.add_argument(
        '--fa_data_list',
        required=False,
        default=None,
        help="data list used to test False Alarm"
    )
    parser.add_argument(
        '--losst',
        default='xent',
        help=' \'0,1\', ==> means use gpu_0 and gpu_1; \'cpu\' means use cpu'
    )
    parser.add_argument(
        '--prefix',
        default=None,
        help='Save prefix'
    )
    parser.add_argument(
        '--topk',
        default=1,
        help='topk to trigger keyword'
    )
    args = parser.parse_args()
    return args


class Evaler():
    def __init__(
        self,
        model_arch: str,
        config: dict,
        args: argparse.Namespace
    ):
        # init config info
        config = self.make_test_config(config, args)
        self.data_config = config['test_data_config']
        self.model_config = config['model_config']
        self.exp_name = config['exp_config']['exp_name']
        self.result_savepath = config['exp_config']['exp_dir'] + "/test_result_{}/".format(args.test_ckpt)
        self.topk = int(args.topk)
        if args.prefix:
            self.result_savepath = self.result_savepath + "/{}".format(args.prefix)
        if not os.path.isdir(self.result_savepath):
            os.makedirs(self.result_savepath)
        
        if args.device == 'cpu':
            device = 'cpu'
            self.master_device = 'cpu'
        else:
            device = [int(x) for x in args.device.split(",")]
            self.master_device = 'cuda:{}'.format(device[0])
        # load model 
        self.model = model_arch(**self.model_config)
        _, _, _ = self.load_ckpt()

        # make test dataset and test dataloader
        self.data_loaders = {}
        if self.data_config['acc_data_list']:
            acc_data_list = self.data_config['acc_data_list']
            acc_test_list = read_list(acc_data_list)
            acc_test_set = Dataset(self.data_config, acc_test_list)
            acc_loader = DataLoader(
                acc_test_set,
                batch_size=None,
                num_workers=12,
                shuffle=False
            )
        else:
            acc_loader = None

        self.data_loaders.update({'acc_loader': acc_loader})

        if self.data_config['fa_data_list']:
            fa_test_list = self.data_config['fa_data_list']
            fa_test_list = read_list(fa_test_list)
            fa_test_set = Dataset(self.data_config, fa_test_list)
            fa_loader = DataLoader(fa_test_set, batch_size=None, num_workers=2, shuffle=False)
        else:
            fa_loader = None
        self.data_loaders.update({'fa_loader': fa_loader})
    
        if args.keyword2id:
            self.kw2id = make_dict_from_file(args.keyword2id)
            self.id2kw = {int(v): k for k, v in self.kw2id.items()}
    
    def load_ckpt(self):
        ckpt = self.data_config['ckpt']
        print (ckpt)
        ckpt = torch.load(ckpt, map_location='cpu')
        model = ckpt['model']
        if 'cv_loss' in ckpt.keys():
            cv_loss = ckpt['cv_loss']
        else:
            cv_loss = float('inf')
        self.model.load_state_dict(model)
        return 1, 1, cv_loss

    @torch.no_grad()
    def run(self):
        self.model.to(self.master_device)
        self.model.eval()
        eval_info = {}
        for key, loader in self.data_loaders.items():
            if loader == None:
                continue
            one_info = self.eval(loader)
            eval_info[key] = one_info
            top1_writer = open("{}/{}.top1".format(self.result_savepath, key), "w")
            for item in one_info:
                key = item['key']
                score = item['score']
                _, top_idx = compute_topk(score, k=self.topk)
                top_idx = [x.item() for x in top_idx]
                top1_writer.write("{} {} {}\n".format(key, top_idx, [score[t] for t in top_idx]))
                #top1_writer.write("{} {} {} {} {}\n".format(key, top_idx[0], score[top_idx[0]], top_idx[1], score[top_idx[1]]))
                top1_writer.flush()
            top1_writer.close()
            
        #num_acc, total_num, error_idx = self.eval_acc(scores, targets)
        #acc = (num_acc / total_num)*100

        #acc_info = {'info': eval_info, 'top1_acc':acc} 
        torch.save(eval_info, '{}/test.result.pt'.format(self.result_savepath))

    @torch.no_grad()
    def eval(self, loader):
        keys = []
        eval_info = []
        for batch_id, data in enumerate(loader):
            if batch_id % 10 == 0:
                print (batch_id)
            utt_keys = [k for k in data[0]]
            keys.extend(utt_keys)
            input_data = [d.to(self.master_device) for d in data[1:]]
            score, target = self.model.evaluate(input_data)
            score = score.cpu().clone().detach()
            target = target.cpu().clone().detach()
            #_, top_idx = compute_topk(score, k=1)
            #for i, k in enumerate(utt_keys):
            #    _, top_idx = compute_topk(score[i], k=1)
            #    print (k, top_idx.item(), score[i][top_idx.item()].item())
            if batch_id == 0:
                scores = score.clone()
                targets = target.clone()
            else:
                scores = torch.cat([scores, score], dim=0)
                targets = torch.cat([targets, target], dim=0) 
        for i, key in enumerate(keys):
            one_result = {
                'key': key, 'score': scores[i], 'target': targets[i]
            }
            eval_info.append(one_result)
        return eval_info

    def eval_acc(self, scores, targets, topK=1):
        sorted_score, top_idx = compute_topk(scores, k=topK)
        num_acc = 0
        total_num = 0
        error_idx = []
        for i, one_targets in enumerate(targets):
            one_top_idx = top_idx[i]
            if one_top_idx.item() == one_targets.item():
                num_acc += 1
            else:
                error_idx.append([i, one_top_idx])
            total_num += 1
        return num_acc, total_num, error_idx
    
    def eval_fa(self, scores, topK=1):
        sorted_score, top_idx = compute_topk(scores, k=topK)
        num_fa = 0
        total_num = 0
        for i, one_top_idx in enumerate(top_idx):
            one_top_idx = one_top_idx[i]
            if one_top_idx.item() in self.id2keyword:
                num_fa += 1
            total_num += 1
        return num_fa, total_num

    def eval_eer(self, scores, targets, exp_name=None):
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        eer, eer_info = compute_eer_skleanr(targets, scores)
        eer_th, fpr, fnr = eer_info
        return eer, eer_info

    def make_test_config(self, config, args):
        exp_config = config['exp_config']
        exp_dir = exp_config['exp_dir']
        exp_name = exp_config['exp_name']
        test_data_config = config['test_data_config']
        n_ckpt = args.test_ckpt
        batch_size = int(args.batch_size)
        ckpt = "{}/{}_{}.pt".format(exp_dir, exp_name, n_ckpt)
        if 'self_crupt' in test_data_config:
            test_data_config.pop('self_crupt') 
        if 'wav_augment' in test_data_config['sph_config']:
            test_data_config['sph_config'].pop('wav_augment') 
        if 'mix_config' in test_data_config:
            test_data_config.pop('mix_config')
        if 'none_target_crupt' in test_data_config:
            test_data_config.pop('none_target_crupt')
        if 'self_corruption' in test_data_config['sph_config']:
            test_data_config['sph_config'].pop('self_corruption')
        if 'none_target_corruption' in test_data_config['sph_config']:
            test_data_config['sph_config'].pop('none_target_corruption')
        if 'trim_config' in test_data_config['sph_config']:
            if 'trim_dither' in test_data_config['sph_config']['trim_config']:
                test_data_config['sph_config']['trim_config'].pop('trim_dither')
            if 'dither' in test_data_config['sph_config']['trim_config']:
                test_data_config['sph_config']['trim_config'].pop('dither')
            if 'trim_type' in test_data_config['sph_config']['trim_config']:
                test_data_config['sph_config']['trim_config'].pop('trim_type')
        test_data_config.update({
            'ckpt': ckpt,
            'shuffle': False,
            'batch_size': batch_size,
            'fetch_key': 'key,speech,word_keyword',
            'acc_data_list': args.acc_data_list,
            'fa_data_list': args.fa_data_list,
        })
        print (test_data_config)
        return config

if __name__ == '__main__':

    args = get_args()
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    from model import m_dict
    config = args.config_file
    config = yaml.load(open(config), Loader=yaml.FullLoader)
    model_arch = config['model_arch']
    model = m_dict[model_arch] 
    eval = Evaler(model, config, args)
    eval.run()
