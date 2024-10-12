# ref: wenet dataset.py
#from multiprocessing.dummy import Process
import torch
import random
import copy
import data.loader.factory as utils
import torch.distributed as dist 

from local.utils import read_list
from torch.utils.data import IterableDataset

class Processer(IterableDataset):
    def __init__(
        self,
        source,
        f,
        *args,
        **kw
    ):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw
    
    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        assert self.source is not None
        return self.f((iter(self.source)), *self.args, **self.kw)

    #def apply(self, f):
    #    return Processer(self, f, *self.args, **self.kw)

class DistributedSampler:
    def __init__(
        self,
        shuffle=True,
        partition=True
    ):
        self.epoch=-1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id 
            self.num_workers = worker_info.num_workers

        return dict(
            rank = self.rank,
            world_size = self.world_size,
            worker_id = self.worker_id,
            num_workers=self.num_workers
        )

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def sample(self, indexes):
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(indexes)
            indexes = indexes[self.rank::self.world_size]
        indexes = indexes[self.worker_id::self.num_workers]
        return indexes

class DataList(IterableDataset):
    def __init__(
        self,
        lists,
        self_corruption=False,
        none_target_corruption_list=None,
        rirs_list=None,
        shuffle=True,
        partition=True,
    ):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)
        self.self_corruption = self_corruption
        if none_target_corruption_list != None:
            self.none_target_corruption_list = none_target_corruption_list
            self.num_none_target_corruption = len(self.none_target_corruption_list)
            self.none_target_corruption = True
        else:
            self.none_target_corruption = False
        if rirs_list != None:
            self.rirs_list = rirs_list
            self.reverb = True
        else:
            self.reverb = False
    
    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
    
    def padding(self, org_list, target_len):
        org_len = len(org_list)
        assert org_len < target_len
        num_repeat = target_len // org_len
        new_list = copy.deepcopy(org_list)
        for x in range(num_repeat):
            new_list += org_list
        return new_list

    def make_corrupt_candidate(self, lists, indexes, num_candidate=5):
        idx = random.choices(indexes, k=num_candidate)
        candidate = []
        for i in idx:
            candidate.append(lists[i])
        if num_candidate == 1:
            candidate = candidate[0]
        return candidate

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = list(range(len(self.lists)))
        indexes = self.sampler.sample(indexes)
        if self.self_corruption:
            self_corrupt_lists = copy.deepcopy(self.lists)
            self_corrupt_indexes = list(range(len(self_corrupt_lists)))
            self_corrupt_indexes = self.sampler.sample(self_corrupt_indexes)
        if self.none_target_corruption:
            none_target_corrupt_lists = self.none_target_corruption_list
            none_target_corrupt_indexes = list(range(len(none_target_corrupt_lists)))
            none_target_corrupt_indexes = self.sampler.sample(none_target_corrupt_indexes)
        if self.reverb:
            rirs_lists = self.rirs_list
            rirs_indexes = list(range(len(rirs_lists)))
            rirs_indexes = self.sampler.sample(rirs_indexes)

        for i, index in enumerate(indexes):
            data = dict(src=self.lists[index], epoch=self.sampler.epoch)
            data.update(sampler_info)
            if self.self_corruption:
                self_corrupt_candidate = self.make_corrupt_candidate(
                    self_corrupt_lists, self_corrupt_indexes
                )
                data.update(self_corruption=copy.deepcopy(self_corrupt_candidate))

                one_neg_candidate = self.make_corrupt_candidate(
                    self_corrupt_lists, self_corrupt_indexes, num_candidate=70
                )
                data.update(neg_candidate=one_neg_candidate)
            if self.none_target_corruption:
                none_target_corrupt_candidate = self.make_corrupt_candidate(
                    none_target_corrupt_lists, none_target_corrupt_indexes
                )
                data.update(none_target_corruption=none_target_corrupt_candidate)
            if self.reverb:
                rirs_src = self.make_corrupt_candidate(rirs_lists, rirs_indexes, num_candidate=1)
                data.update(rirs=rirs_src)
            yield data


def Dataset(
    conf, 
    d_list,
):
    # check speech config
    sph_config = conf.get('sph_config', None)
    if not sph_config:
        raise NotImplementedError(
            "sph_config should be specific, there are no any default config for " + 
            "speech feats"
        )
    data_list_config = {'lists': d_list}
    shuffle = conf.get('shuffle', True)
    data_list_config.update({'shuffle': shuffle})
    corruption_config = {}
    if sph_config.get('self_corruption', False):
        corruption_config.update({'self_corruption': sph_config.get('self_corruption')})
        self_corruption = True 
        data_list_config.update({'self_corruption': self_corruption})

    if sph_config.get('none_target_corruption', False):
        corruption_config.update({'none_target_corruption': sph_config.get('none_target_corruption')})
        none_target_corruption_list = corruption_config['none_target_corruption']['corrupt_list']
        none_target_corruption_list = read_list(none_target_corruption_list)
        data_list_config.update({'none_target_corruption_list': none_target_corruption_list})
    if sph_config.get('rirs', False):
        rirs_list = sph_config['rirs']['rirs_list']
        rirs_list = read_list(rirs_list)
        data_list_config.update({'rirs_list': rirs_list})
        

    # Build data list
    dataset = DataList(**data_list_config)

    # START DATA PROCESS!!!
    dataset = Processer(dataset, utils.process_raw)

    if len(corruption_config) > 0:
        dataset = Processer(dataset, utils.process_corruption, corruption_config)

    # prepare speech feats
    dataset = Processer(dataset, utils.process_speech_feats, sph_config)

    # prepare text feats, such as label, keyword etc.
    text_config = conf.get('text_config', {})
    dataset = Processer(dataset, utils.process_text_feats, **text_config)

    # process keyword setting
    keyword_setting = conf.get('keyword_config', None) 
    if isinstance(keyword_setting, dict):
        keyword_format = keyword_setting.get('format')
        assert keyword_format in ['sample', 'fix', 'test']
        keyword_config = keyword_setting.get('config', {})
        if keyword_format == 'sample':
            crpt_list = copy.deepcopy(d_list)
            random.shuffle(crpt_list)
            keyword_config.update({'neg_len': 70})
            dataset = Processer(dataset, utils.process_sampled_keyword_from_label,  **keyword_config)
        elif keyword_format == 'fix':
            dataset = Processer(dataset, utils.process_fix_keyword, **keyword_config)
        elif keyword_format == 'test':
            pass
        else:
            raise NotImplementedError("keyword format only Support <sample> / <fix_with_segment>")
    elif keyword_setting != None:
        raise NotImplementedError("keyword config should be dict")
    else:
        pass
    
    sot_label_config = conf.get('sot_config', None)
    if sot_label_config:
        dataset = Processer(dataset, utils.process_sot_label, **sot_label_config)

    permuate_label_config = conf.get('permuate_label_config', None)
    if permuate_label_config:
        dataset = Processer(dataset, utils.process_permuate_label, **permuate_label_config)
    
    conditional_chain_config = conf.get('conditional_chain', None)
    print (conditional_chain_config)
    if conditional_chain_config:
        dataset = Processer(dataset, utils.process_conditional_chain_label, **conditional_chain_config)
    #sequentail_label_config = conf.get('sequentail_label_config', None)
    #if sequentail_label_config:
    #    dataset = Processer(dataset, utils.process_sequentail_label, **sequentail_label_config) 
    # process list data 
    dataset = Processer(dataset, utils.process_list_data)

    # process length information
    dataset = Processer(dataset, utils.make_length)

    # make batch
    dataset = Processer(dataset, utils.make_batch, conf.get('batch_size', 256))

    # if fetch_key from config is not None, use that fetch_key
    if conf.get('fetch_key', None):
        fetch_key = conf.get('fetch_key').split(",")
    else:
        fetch_key = ['speech', 'label', 'keyword']

    # fetch tensor
    dataset = Processer(dataset, utils.fetch_tensor, fetch_key)
    return dataset

