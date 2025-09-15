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
from data.loader.data_loader_kw_init_fin import Dataset
from local.utils import WarmUpLR, read_list, Recorder
from torch.utils.tensorboard import SummaryWriter
import shutil
import sys
import json, os, tempfile
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help="config file in yaml format e.g. config/ref.yaml"
    )
    parser.add_argument(
        '--step',
        required=False,
        default=0,
        type=int,
        help="step"
    )
    parser.add_argument(
        '--world_size',
        required=True,
        type=int,
        help="world size required by torch.distributed usually set it as 4"
    )
    parser.add_argument(
        '--rank',
        required=True,
        type=int,
        help="rank"
    )
    parser.add_argument(
        '--port',
        required=False,
        default="1234",
        type=str,
        help="port"
    )
    parser.add_argument(
        '--gpu',
        required=True,
        help='gpu id we used, we are not support train with cpu'
    )
    parser.add_argument(
        '--seed',
        default=2022,
        help="random seed"
    )
    parser.add_argument(
        '--exp_root',
        default='exp/',
        help="experinments root dir"
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume from an existing run_dir (contains run.json). If set, loader will restore from last_checkpoint and continue.'
    )
    parser.add_argument(
        '--start-time',
        type=str,
        default=None,
        help='Specify the start time of each run, e.g. 20230601_101010.'
    )

    args = parser.parse_args()
    return args


class Trainer():
    def __init__(
        self,
        model_arch: str,
        config_file: dict,
        rank: int,
        world_size: int,
        random_seed=2022,
        resume_from=None
    ):
        # init config info
        self.config_file = config_file
        self.rank = rank
        self.seed = random_seed
        self.device = torch.device('cuda')
        self.world_size = world_size
        self.resume_from = resume_from
        # continue training from break point
        if resume_from:
            run_dir = resume_from
            self.run_dir_name = os.path.basename(os.path.normpath(run_dir))
            self.run_dir = run_dir
            config_dir = os.path.join(run_dir, 'configs')
            self.model_config = yaml.load(open(os.path.join(config_dir, 'model.yaml')), Loader=yaml.FullLoader)
            self.data_config = yaml.load(open(os.path.join(config_dir, 'data.yaml')), Loader=yaml.FullLoader)
            self.exp_config = yaml.load(open(os.path.join(config_dir, 'exp.yaml')), Loader=yaml.FullLoader)
            # --- support extending epochs on resume via the current config file ---
            # If the provided config (this run's --config) contains data_config.epoch
            # and it is GREATER than the saved one, use the larger (extend only).
            self._epoch_override_requested = None
            try:
                override_epoch = int(config_file.get('data_config', {}).get('epoch', None))
            except Exception:
                override_epoch = None
            if override_epoch is not None:
                prev_epoch = int(self.data_config.get('epoch', 0))
                if override_epoch > prev_epoch:
                    # apply extension
                    self.data_config['epoch'] = override_epoch
                    self._epoch_override_requested = {
                        "previous": prev_epoch,
                        "requested": override_epoch,
                        "applied": override_epoch
                    }
                else:
                    # do not shrink; keep previous
                    self._epoch_override_requested = {
                        "previous": prev_epoch,
                        "requested": override_epoch,
                        "applied": prev_epoch
                    }
            # --- end: epoch extension support ---
        else:
            self.model_config = config_file['model_config']
            self.data_config = config_file['data_config']
            self.exp_config = config_file['exp_config']
            # Build a unique run directory name
            self.ts = args.start_time
            slug = self.exp_config.get('run_slug', '') or os.environ.get('RUN_SLUG', '')
            # sanitize slug: keep alphanum, dash, underscore only
            if slug:
                slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(slug)).strip("-")
            self.run_dir_name = f"run_{self.ts}" + (f"_{slug}" if slug else "")
            self.run_dir = os.path.join(self.exp_config['exp_dir'], self.run_dir_name)
        self.model = model_arch(**self.model_config)

        if not resume_from:
            if self.rank == 0:
                try:
                    os.makedirs(self.run_dir, exist_ok=False)
                except FileExistsError:
                    # Extremely unlikely due to timestamp; fall back to a counter
                    i = 1
                    while True:
                        alt = os.path.join(self.exp_config['exp_dir'], f"{self.run_dir_name}_{i}")
                        try:
                            os.makedirs(alt, exist_ok=False)
                            self.run_dir = alt
                            break
                        except FileExistsError:
                            i += 1
            # wait for rank 0 to create dir
            else:
                while not os.path.exists(self.run_dir):
                    pass

        self.exp_config['log_config']['filename'] = "{}/train.{}.log".format(
            self.run_dir,
            self.rank
        )
        # init recorder        
        self.recorder = Recorder(self.exp_config, self.run_dir) 

    def _read_last_epoch_from_run(self):
        try:
            path = os.path.join(self.run_dir, 'run.json')
            with open(path, 'r') as f:
                meta = json.load(f)
            return meta.get('training_state', {}).get('last_epoch', None)
        except Exception:
            return None


    def backup_configs(self):
        for k, v in self.config_file.items():
            self.recorder.info("{} config : {}".format(k.upper(), v))
        
        if (self.rank == 0): 
            config_dir = os.path.join(self.run_dir, 'configs')
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            # model config backup in expdir
            self.model_config_file = os.path.join(config_dir, 'model.yaml')
            yaml.dump(self.model_config, open(self.model_config_file, 'w'))
            # data config backup in expdir
            self.data_config_file = os.path.join(config_dir, 'data.yaml')
            yaml.dump(self.data_config, open(self.data_config_file, 'w'))
            # exp config backup in expdir
            self.exp_config_file = os.path.join(config_dir, 'exp.yaml')
            yaml.dump(self.exp_config, open(self.exp_config_file, 'w'))
            # data process code backup in expdir
            self.data_dir = os.path.join(self.run_dir, 'data')
            if os.path.exists(self.data_dir):
                shutil.rmtree(self.data_dir)
            shutil.copytree("data", self.data_dir, symlinks=True)
            # model structure code backup in expdir
            self.model_dir = os.path.join(self.run_dir, 'model')
            if os.path.exists(self.model_dir):
                shutil.rmtree(self.model_dir)
            shutil.copytree("model", self.model_dir, symlinks=True)
    

    def save_meta(self, cmd, cwd):
        """
        Create a unique run directory under exp_config['exp_dir'] on rank==0
        and save training-related metadata before training starts.

        Directory name format: run_{YYYYMMDD_HHMMSS}_{custom_slug}
        - custom_slug is optional and taken from exp_config['run_slug'] if provided.

        Files written:
          - run.json : consolidated training-related info (configs, env, etc.)
          - COMMIT_ID : current git commit id (if available)
          - env.lock : environment lock produced by tools/env_lock.py
        """
        
        if self.rank != 0:
            return
        import os, json, subprocess, sys, socket

        # Try to get current git commit id
        commit_id = "UNKNOWN"
        commit_id_file = os.path.join(self.run_dir, "COMMIT_ID")
        try:
            commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode().strip()
        except Exception:
            pass
        try:
            with open(commit_id_file, "w") as f:
                f.write(f"{commit_id}\n")
        except Exception:
            # Non-fatal
            pass

        # Produce env.lock by invoking tools/env_lock.py
        from tools.env_lock import write_env_lock
        env_dict = write_env_lock(self.run_dir)

        # Consolidate run metadata
        meta = {
            "command": cmd,
            "cwd": cwd,
            "run_dir": self.run_dir,
            "run_dir_name": self.run_dir_name,
            "start_time": self.ts,
            "commit_id": commit_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "device": str(self.device),
            "random_seed": self.seed,
            "hostname": socket.gethostname(),
            "python_version": sys.version,
            "torch_version": getattr(__import__('torch'), '__version__', 'unknown'),
            "paths": {
                "data_config": self.data_config_file,
                "exp_config": self.exp_config_file,
                "model_config": self.model_config_file,
                "data_dir": self.data_dir,
                "model_dir": self.model_dir,
                "commit_id": commit_id_file
            },
            "config": self.config_file,
        }

        try:
            with open(os.path.join(self.run_dir, "run.json"), "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # Expose the path for later use if needed
        self.run_dir = self.run_dir

    # import json, os, tempfile

    def _update_run_json_atomic(self, patch: dict):
        if self.rank != 0:
            return
        path = os.path.join(self.run_dir, "run.json")
        try:
            with open(path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

        # 浅层合并：顶层键直接覆盖；嵌套你可以按需细化
        for k, v in patch.items():
            meta[k] = v

        # 原子写：写到临时文件再替换
        d = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(prefix="runjson_", dir=d)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass

    def compute_redundancy(self, n):
        r1 = n % self.world_size
        r2 = ((n - r1) / self.world_size) % self.batch_size
        rt = n - r1 - r2 * self.world_size
        return int(rt)

    def make_data_loader(self):
        # parse datalist
        data_list_file = self.data_config['data_list']
        self.batch_size = self.data_config['batch_size']
        cv_list_file = self.data_config.get('valid_list', None)

        if cv_list_file:
            cv_list = read_list(cv_list_file, split_cv=False, shuffle=True)
            tr_list = read_list(data_list_file, split_cv=False, shuffle=True)
        else:
            tr_list, cv_list = read_list(data_list_file, split_cv=True, shuffle=True)
        num_train_sample = len(tr_list)
        num_valid_sample = len(cv_list)

        rt_train_sampple = self.compute_redundancy(num_train_sample)
        rt_cv_sample = self.compute_redundancy(num_valid_sample)

        tr_list = tr_list[:rt_train_sampple]
        cv_list = cv_list[:rt_cv_sample]

        if self.data_config.get('egs_format', False):
            egs_path = os.path.dirname(data_list_file)
            assert os.path.isfile("{}/train.samples".format(egs_path))
            with open("{}/train.samples".format(egs_path),'r') as ef:
                self.num_samples = int(ef.readline().strip())
        else:
            self.num_samples = len(tr_list)
        num_worker = self.data_config.get('num_worker', 10)

        self.tr_set = Dataset(
            self.data_config,
            tr_list,
        )
        self.cv_set = Dataset(
            self.data_config,
            cv_list,
        )

        self.tr_loader = DataLoader(
            self.tr_set,
            batch_size=None,
            num_workers=num_worker
        )
        self.cv_loader = DataLoader(
            self.cv_set,
            batch_size=None,
            num_workers=3
        )

        if self.data_config['start_epoch'] == 0:
            self.recorder.info(
                "Num Training samples: {}, Num Valid samples: {}".format(len(tr_list), len(cv_list))
            )
            self.recorder.info(
                "Num Worker: {}".format(num_worker)
            )

    def init_opt_model(self):
        # warm_up setting: compute update steps per epoch
        self.batch_size = self.data_config['batch_size']
        world_size = self.world_size
        steps_per_epoch = self.num_samples // (self.batch_size * world_size)
        steps_per_epoch = 1 if steps_per_epoch == 0 else steps_per_epoch
        if self.exp_config.get('warm_up_peak_epoch', False):
            warm_up_peak_epoch = self.exp_config.get('warm_up_peak_epoch')
        else:
            warm_up_peak_epoch = 5
        warm_up_peak_step = warm_up_peak_epoch * steps_per_epoch
        # init model & optim & dist
        num_param = sum([v.numel() for v in self.model.parameters()])
        self.clip_value = self.exp_config.get('clip_value', 10.0)
        self.optim = optim.Adam(
            self.model.parameters(), 
            **self.exp_config['optim_config']
        )

        # Decide tensorboard directory under current run
        tensorboard_dir = os.path.join(self.run_dir, 'logs', 'tensorboard')

        # Decide resume vs scratch based on --resume-from
        if self.resume_from:
            last_epoch = self._read_last_epoch_from_run()
            if last_epoch is None:
                raise FileNotFoundError(
                    f"--resume-from given but {os.path.join(self.run_dir,'run.json')} lacks training_state.last_epoch"
                )
            # Load checkpoint from last_epoch and continue
            ckpt = self.load_endpoint(last_epoch)
            self.global_step = self.load_ckpt(ckpt)
            start_epoch = int(last_epoch) + 1
            self.data_config['start_epoch'] = start_epoch
            if self.rank == 0:
                self.tb_writer_train = SummaryWriter(tensorboard_dir, filename_suffix='train', purge_step=self.global_step)
                self.tb_writer_cv = SummaryWriter(tensorboard_dir, filename_suffix='cv', purge_step=start_epoch)
            self.recorder.info(f"Resume training from epoch: {start_epoch} (loaded last_epoch={last_epoch})")
            # record any epoch override coming from the current config
            if getattr(self, "_epoch_override_requested", None) is not None and self.rank == 0:
                ov = self._epoch_override_requested
                self.recorder.info(f"[resume] total epochs: saved={ov['previous']}, requested={ov['requested']}, effective={self.data_config['epoch']}")
                self._update_run_json_atomic({
                    "resume_overrides": {
                        "epoch": ov
                    }
                })
        else:
            # From scratch / finetune: start at 0
            start_epoch = 0
            self.data_config['start_epoch'] = 0
            self.global_step = 0
            if self.exp_config.get('finetune_config', False):
                self.recorder.info("Finetune from existing checkpoint {}".format(
                    self.exp_config['finetune_config']['trained_ckpt']
                ))
                self.load_ckpt_part(self.exp_config['finetune_config']['trained_ckpt'])
            if self.rank == 0:
                self.tb_writer_train = SummaryWriter(tensorboard_dir, filename_suffix='train')
                self.tb_writer_cv = SummaryWriter(tensorboard_dir, filename_suffix='cv')
            self.recorder.info("Start training from scratch (or finetune) at epoch 0")

        self.scheduler = WarmUpLR(self.optim, warmup_steps=warm_up_peak_step)
        self.scheduler.set_step(self.global_step)
        # if self.exp_config.get('finetune', False):
        #     finetune_config = self.exp_config.get('finetune')
        #     self.init_from_trained(**finetune_config)

        # init distributed training
        if self.world_size > 1:
            dist.init_process_group(
                'nccl', #TODO: check why dragon03 can not use nccl
                world_size=self.world_size,
                rank=self.rank
            )
            self.model.cuda()
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model
            )
        else:
            self.model.to(self.device)

        if start_epoch == 0:
            self.recorder.info("WarmUp epoch: {} WarmUp steps:{}".format(warm_up_peak_epoch, warm_up_peak_step))
            self.recorder.info("Gradient Clip Value: {}".format(self.clip_value))
            self.recorder.info("Number parameter: {}".format(num_param))
        else:
            self.recorder.info("Continue training from epoch: {}".format(start_epoch))

    # percentage_fix_layer: mean how many layer in """NOTE: TRAINED CKPT""" will be fixed
    # NOTE: THAT the percentage means layer in trained ckpt not in new ckpt
    def init_from_trained(self, trained_ckpt, percentage_fix_layer='last'):
        self.recorder.info(
                "Init from trained checkpoint {} {}\% of them will be fixed".format(trained_ckpt, percentage_fix_layer)
        )
        trained_ckpt = torch.load(trained_ckpt, map_location='cpu')
        trained_model = trained_ckpt['model']
        trained_keys = list(trained_model.keys())
        fix_component = []
        current_state_dict = self.model.state_dict()
        num_trained_layers = len(trained_keys)
        if isinstance(percentage_fix_layer, int) and (percentage_fix_layer > 1):
            percentage_fix_layer = float(percentage_fix_layer) / 100
        num_fix_layer = num_trained_layers - 2 if percentage_fix_layer == 'last' else float(percentage_fix_layer) * num_trained_layers
        self.recorder.info("load parameters from trained model: {}".format(trained_ckpt.keys()))
        for i, k in enumerate(trained_keys):
            traiend_param = trained_model[k]
            if (current_state_dict[k].size() == traiend_param.size()) and (i < num_fix_layer):
                fix_component.append(k)
            else:
                trained_model.pop(k)
        self.model.load_state_dict(trained_model, strict=False)
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if name in fix_component:
                param.requires_grad = False
            else:
                self.recorder.info("{} are trainable".format(name))
        
    def load_ckpt(self, ckpt):
        ckpt_dict = torch.load(ckpt, map_location='cpu')
        model = ckpt_dict['model']
        opt = ckpt_dict['opt']
        step = ckpt_dict['step']

        self.optim.load_state_dict(opt)
        for state in self.optim.state.values():
            for k, v in state.items():
                if k == 'step':
                    continue
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.model.load_state_dict(model)
        return step

    def need_to_load(self, key, pretrained_module_prefix):
        for module_prefix in pretrained_module_prefix:
            if key.startswith(module_prefix):
                return True
        #if key.startswith('au_'):
        #    return True
        #if key.startswith('det_net'):
        #    return True
        return False

    def load_ckpt_part(self, ckpt):
        ckpt_dict = torch.load(ckpt, map_location='cpu')
        model_dict = self.model.state_dict()
        pretrained_dict = ckpt_dict['model']
        pretrained_module_prefix = self.exp_config['finetune_config']['pretrained_module_prefix']
        filtered_dict = {k: v for k, v in pretrained_dict.items() if self.need_to_load(k, pretrained_module_prefix)}
        model_dict.update(filtered_dict)
        self.model.load_state_dict(model_dict)

    @torch.no_grad()
    def cross_valid(self):
        cv_model = copy.deepcopy(self.model)
        cv_model.eval()
        cv_detail_loss = {'total_loss': 0}
        num_utt = 0
        for batch_id, cv_data in enumerate(self.cv_loader):
            n = cv_data[0].size(0)
            cv_data = (d.to(self.device) for d in cv_data)
            num_utt += n
            total_loss, detail_loss = cv_model(cv_data) 
            detail_loss = self.detach_from_graph(detail_loss)
            cv_detail_loss['total_loss'] += (self.detach_from_graph(total_loss) * n)
            #detail_loss['total_loss'] = total_loss*n

            for key, value in detail_loss.items():
                if key not in cv_detail_loss.keys():
                    cv_detail_loss[key] = value*n
                else:
                    cv_detail_loss[key] += (value*n)

        cv_detail_loss = {
            key : value/num_utt for key, value in cv_detail_loss.items()
        }
        return cv_detail_loss
    
    def detach_from_graph(self, para):
        if isinstance(para, torch.Tensor):
            para = para.detach().clone()
        if isinstance(para, dict):
            para = {k: v.detach().clone() for k,v in para.items()}
        return para

    def detach_state_dict(self):
        d_model = copy.deepcopy(self.model)
        opt = copy.deepcopy(self.optim)
        if isinstance(d_model, torch.nn.parallel.DistributedDataParallel):
            d_model = d_model.module.state_dict()
        else:
            d_model = d_model.state_dict()
        opt = opt.state_dict()
        return d_model, opt

    def record_step(self, r_loss):
        assert (isinstance(r_loss, dict))
        for key, value in r_loss.items():
            assert(key in ['train', 'cv'])
            self.recorder.record_detail(
                value, self.epoch, self.global_step,
                model=None, opt=None, tag=key
            )
            for loss_key, loss_value in value.items():
                self.tb_writer_train.add_scalar('{}/{}'.format(loss_key, key), loss_value, self.global_step)
    
    def record_epoch(self, cv_loss=None):
        model, opt = self.detach_state_dict()
        self.recorder.record_epoch(
            self.epoch, self.global_step,
            model, opt, cv_loss
        )
        for loss_key, loss_value in cv_loss.items():
            self.tb_writer_cv.add_scalar('{}/{}'.format(loss_key, 'cv'), loss_value, self.epoch)
        self._update_run_json_atomic({
            "training_state": {
                "last_epoch": self.epoch,
                "global_step": self.global_step
            }
        })

    def load_endpoint(self, epoch):
        # exp_dir = self.exp_config['exp_dir']
        exp_name = self.exp_config['exp_name']
        run_dir = self.run_dir
        ckpt = "{}/{}_{}.pt".format(run_dir, exp_name, epoch)
        # ckpt = "{}/{}_{}.pt".format(exp_dir, exp_name, epoch)
        if os.path.isfile(ckpt):
            return ckpt
        else:
            raise FileNotFoundError(
                "{} does not exits check the start epoch from yaml config file".format(ckpt)
            )
    
    def avg_model(self):
        # average the last 10 model
        max_epoch = self.data_config['epoch']
        avg_epoch = self.exp_config.get('avg_epoch', 10)
        #max_epoch = 50
        min_epoch = max_epoch - avg_epoch
        valid_ckpt = {k:0 for k in range(min_epoch, max_epoch)}
        valid_loss = []
        for e in range(min_epoch, max_epoch):
            ckpt = "{}/{}_{}.pt".format(self.run_dir, self.exp_config['exp_name'],e)
            ckpt = torch.load(ckpt, map_location='cpu')
            one_valid_loss = ckpt['cv_loss']['total_loss'].item()
            valid_ckpt[e] = ckpt['model']
            valid_loss.append(one_valid_loss)
        sort_idx = sorted(range(len(valid_loss)), key=lambda k: valid_loss[k])
        min_idx = sort_idx[:10]
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
                avg_model[k] = torch.true_divide(avg_model[k], 10)
        self.recorder.save_state(avg_model, epoch='avg')
        
    def train(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.model.to(self.device)
        self.model.train()
        start_epoch = self.data_config['start_epoch']
        end_epoch = self.data_config['epoch']
        self.recorder.info("Start training the log is written in {}".format(
                self.exp_config['log_config']['filename']
            )
        )
        if start_epoch >= end_epoch:
            self.recorder.info(f"No training to run: start_epoch {start_epoch} >= end_epoch {end_epoch}.")
            return
        #TODO: add model.join context for distributed data parallel
        for epoch in range(start_epoch, end_epoch):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.tr_set.set_epoch(epoch)
            for batch_id, data in enumerate(self.tr_loader):
                torch.cuda.empty_cache()
                clr = self.optim.param_groups[0]['lr']
                tr_record_dict = {}
                tr_record_dict['lr']=clr
                b = data[0].size(0)
                #if b != self.batch_size:
                #    continue
                train_data = (d.to(self.device) for d in data)
                loss, detail_loss = self.model(train_data)
                if loss is None:
                    print("Skip invalid batch")
                    continue
                self.optim.zero_grad()
                loss.backward()
                grad_norm = clip_grad_norm_(self.model.parameters(), 5)
                if torch.isfinite(grad_norm):
                    self.optim.step()
                else:
                    self.recorder.info("!!! INFINITE grad in epoch: {}, batch_id: {}".format(
                            epoch, batch_id
                        )
                    )
                self.scheduler.step()
                self.global_step += 1
                tr_record_dict['total_loss'] = loss
                tr_record_dict.update(detail_loss)
                if self.rank == 0:
                    self.record_step({'train': tr_record_dict})

            cv_record_dict = self.cross_valid()
            if self.rank == 0:
                self.record_step({'cv': cv_record_dict})
                self.record_epoch(cv_record_dict)
        
    def run(self, step, cmd, cwd):

        # train step
        if step <= 0:
            if not self.resume_from:
                self.backup_configs()
                self.save_meta(cmd, cwd)
            else:
                self.recorder.info(f"Resume from {self.resume_from}")

        if step <= 1:
            self.make_data_loader()
            self.init_opt_model()
            self.train()

        # avg model step
        if (step <= 2) and (self.rank == 0):
            self.avg_model()
        
        # evaluate ...


if __name__ == '__main__':
    args = get_args()
    launch_cmd = " ".join(sys.argv)
    cwd = os.getcwd()
    #this line support load yaml config file recursively e.g.
    # config.yaml
    # item1: value1
    # item2: !include "config2.yaml"
    # YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['MASTER_ADDR'] = '127.0.0.1' # only support gloo in single machine with multi-deivce
                                            # multi-machine with mulit-device should change this param
                                            # as the correct ip 
    os.environ['MASTER_PORT'] = args.port # check whether this port has been occupied.
    config = yaml.load(open(args.config),Loader=yaml.FullLoader)
    from model import m_dict
    model_arch = config['model_arch']
    model = m_dict[model_arch]
    #torch.multiprocessing.set_sharing_strategy('shared_memory')
    trainer = Trainer(
        model, config, world_size=args.world_size, rank=args.rank,
        resume_from=args.resume_from
    )
    trainer.run(args.step, launch_cmd, cwd)
