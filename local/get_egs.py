import os
import random
import sys
import json
import torch
import torchaudio
import copy
import argparse
import multiprocessing
from tqdm import tqdm
from utils import make_dict_from_file

#Letter and digit
LD = '0123456789abcdefghijklmnopqrstuvwxyz'

class Worker(multiprocessing.Process):
    def __init__(
            self, job_plan, pbar, index, path, prefix, state_dict
    ):
        super(Worker, self).__init__()
        self.plan = job_plan
        self.pbar = pbar
        self.sample_rate = 16000
        self.state_dict = state_dict
        self.target = "{}/{}.egs.{}.pt".format(path, prefix, index+1)

    def run(self):
        for i, one_job in enumerate(self.plan):
            self.pbar.update(1)
            if 'sph' not in one_job:
                raise KeyError("<sph> not in list")
            wav, sr = torchaudio.load(one_job['sph'])
            if sr != self.sample_rate:
                print ("{} {} sample rate not {} !!".format()) 
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(wav)
            one_job.update({'sph': wav})
        #self.state_dict[self.name] = self.plan
        torch.save(self.plan, self.target)
    
class Egs(argparse.ArgumentParser):
    def __init__(self):
        super(Egs, self).__init__()
        self.add_argument('--datalist', required=True)
        self.add_argument('--dest', required=True)
        self.add_argument('--utt2dur', required=False, default=None)
        self.add_argument('--sample_rate', required=False, default=16000)
        self.add_argument('--bits_per_sample', required=False, default=16)
        self.add_argument(
            '--prefix', 
            required=True, 
            default='train',
            help='save prefix such as: train, valid, test, none_target_corrupt[augment noise]'
        )
        self.add_argument(
            '--MegaBytePerEgs', 
            type=float,
            required=False, default=1000
        )
        self.add_argument(
            '--num_worker', 
            type=int,
            required=False, default=10
        )
        self.parse_args()
        if not os.path.isdir(self.dest):
            os.makedirs(self.dest)
        self.init_param()
    
    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            msg = ('unrecongized arguments %s')
            self.error(msg % " ".join(argv))
        args_dict = args.__dict__
        for key, value in args_dict.items():
            setattr(self, key, value)

    def init_param(self):
        if self.utt2dur:
            self.utt2dur = make_dict_from_file(self.utt2dur)
            self.utt2dur = {k: float(v) for k, v in self.utt2dur.items()}
        if not os.path.isdir(self.dest):
            os.makedirs(self.dest)
        self.abs_path = os.path.abspath(self.dest)
        assert os.path.isfile(self.datalist)
        self.data_list = dict()
        jf = open(self.datalist, 'r')
        self.num_drop = 0
        for line in jf.readlines():
            line = line.strip()
            try:
                obj = json.loads(line)
            except:
                try:
                    key, wav = line.split(" ")
                    obj = {'key': key, 'sph': wav}
                except:
                    raise TypeError("data list should be json or <uttid uttpath> format")
            key = obj['key']
            if self.utt2dur:
                if key not in self.utt2dur:
                    self.num_drop += 1
                    continue
                dur = self.utt2dur[key]
            else:
                dur = 2
            num_bits = int(dur * self.sample_rate * self.bits_per_sample)
            # torchaudio.load will load wav data with normalizations, 
            # so that int -> float && sizeof(float) / sizeof(int) = 2
            wav_mb = (num_bits / 8 / 1024 / 1024)*2 
            obj.update({'size': wav_mb}) 
            if key in self.data_list:
                rand_suffix = random.choices(LD, k=10)
                key = "{}_{}".format(key, ''.join(rand_suffix))
            self.data_list.update({key: obj})

    def run(self):
        job_plan = []
        one_plan = []
        current_megabyte = 0
        pbar_dict = {}
        state_dict = multiprocessing.Manager().dict()
        num_samples = len(self.data_list)
        for key, value in self.data_list.items():
            if current_megabyte > self.MegaBytePerEgs:
                job_plan.append(copy.deepcopy(one_plan))
                current_megabyte = 0
                one_plan = []
            current_megabyte += value['size']
            value.pop('size')
            one_plan.append(value)
        job_plan.append(one_plan)
        num_egs = len(job_plan)
        jobs = [] 
        for i, plan in enumerate(job_plan):
            one_pbar = tqdm(total=len(plan), desc='Worker: {}'.format(i), ascii=True)
            pbar_dict[i] = one_pbar
            one_worker = Worker(copy.deepcopy(plan), one_pbar, i, self.dest, self.prefix, state_dict)
            #pbar_dict[i] = one_pbar
            jobs.append(one_worker)
        
        num_worker = self.num_worker if len(jobs) > self.num_worker else len(jobs)

        workers = {x:jobs[x] for x in range(num_worker)}

        for worker in workers.values():
            worker.start()
            jobs.remove(worker) 
        
        while (len(jobs) > 0) or (any(w.is_alive() for w in workers.values())):
            for wid in workers.keys():
                if len(jobs) == 0:
                    continue
                if not workers[wid].is_alive():
                    #jobs.remove(workers[wid])
                    one_job = random.choice(jobs)
                    workers[wid] = one_job
                    workers[wid].start()
                    jobs.remove(one_job)
                
        for pbar in pbar_dict.values():
            pbar.close() 
        with open('{}/{}.list'.format(self.dest, self.prefix), 'w') as tf:
            for x in range(num_egs):
                tf.write("{}/{}.egs.{}.pt\n".format(self.abs_path, self.prefix, x + 1))

        with open('{}/{}.samples'.format(self.dest, self.prefix), 'w') as tf:
            tf.write("{}\n".format(num_samples))
        
        
def main():
    egs = Egs()
    egs.run()
if __name__ == '__main__':
    main()