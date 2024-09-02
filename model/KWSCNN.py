import torch
import torch.nn as nn
import torch.nn.functional as F
from model.NetModules import BaseConv, BaseFNN, make_mix_target

class ConvBlock(BaseConv):
    def __init__(
        self,
        i_channel, o_channel,
        kernel_size, stride,
        norm, input_dim=None,
        padding=0, 
        act='ReLU',
        norm_before=False,
        pooling=None
    ):
        super().__init__(
            i_channel=i_channel, o_channel=o_channel,
            conv=nn.Conv2d, kernel_size=kernel_size, stride=stride,
            norm=norm, input_dim=input_dim, act=act,padding=padding,
            norm_before=norm_before, pooling=pooling
        )

class KWSCNN(nn.Module):
    def __init__(
        self,
        base_settings,
        conv_settings,
        mlp_settings,
        bn_settings,
        homo_weight=1,
        homo_pos=None,
        pfeats='clean', ptarget='clean',
        task='base',
        loss='xent',
        mt=True
    ):
        super(KWSCNN, self).__init__()
        self.conv_blocks = nn.ModuleList()
        self.mlp_blcoks = nn.ModuleList()
        num_class = base_settings.get('num_class', 36)
        input_dim = base_settings.get('input_dim', 80)
        conv_channels = conv_settings.get('conv_channels', [32, 64, 128, 64, 128, 256, 512])
        conv_kernels = conv_settings.get('conv_kernels', 3) 
        conv_strides = conv_settings.get('conv_strides', 2)
        conv_module = conv_settings.get('conv_module', None)
        norm_layer = conv_settings.get('norm_layer', len(conv_channels))
        num_mlp_block = mlp_settings.get('num_mlp_block', None)
        mlp_module = mlp_settings.get('mlp_module', None)
        bn_pooling = bn_settings.get('bn_pooling', 'avg')
        bottle_neck = bn_settings.get('bottle_neck_dim', None)
        if num_mlp_block == None:
            self.mlp_blcoks.append(nn.Identity())
        ci_channels = [1] + conv_channels[:-1]
        co_channels = conv_channels
        ci_channels = zip(ci_channels, co_channels)
        feats_dim = input_dim
        time_dim = 98
        for x, (ic, oc) in enumerate(ci_channels):
            one_conv_setting = {}
            one_conv_setting.update(**conv_module)
            if isinstance(conv_kernels, list):
                one_conv_setting.update({'kernel_size': conv_kernels[x]})
            else:
                one_conv_setting.update({'kernel_size': conv_kernels})

            if isinstance(conv_strides, list):
                one_conv_setting.update({'stride': conv_strides[x]})
            else:
                one_conv_setting.update({'stride': conv_strides})
            one_conv_setting.update({'input_dim': [time_dim, feats_dim]})
            #if x > 2:
            if x >= norm_layer:
                print ("Mute {}".format(x))
                one_conv_setting.update({'norm': 'id'})
            one_conv_block = ConvBlock(
                i_channel=ic, o_channel=oc, **one_conv_setting
            )
            self.conv_blocks.append(one_conv_block)
            feats_dim = one_conv_block.d
            time_dim = one_conv_block.t

        if bn_pooling == 'avg':
            self.pooling = nn.AdaptiveAvgPool3d((conv_channels[-1], 2,2))
            self.pooling_act = nn.ReLU()
            cnov_outdim = conv_channels[-1]*2*2
        elif bn_pooling == 'max':
            self.pooling = nn.AdaptiveMaxPool3d((conv_channels[-1], 2,2))
            self.pooling_act = nn.ReLU()
            cnov_outdim = conv_channels[-1]*2*2
        else:
            self.pooling = nn.Identity()
            self.pooling_act = nn.Identity()
            cnov_outdim = feats_dim * conv_channels[-1]
        
        if bottle_neck:
            assert(isinstance(bottle_neck, int))
            self.bottle_neck = nn.Linear(cnov_outdim, bottle_neck)
        else:
            self.bottle_neck = nn.Identity()
        
        if num_mlp_block != None:
            one_mlp_setting = {}
            one_mlp_setting.update(**mlp_module)
            assert(isinstance(num_mlp_block, int))
            for x in range(num_mlp_block):
                if x == 0:
                    indim = bottle_neck if bottle_neck else cnov_outdim
                else:
                    indim = feats_dim
                one_mlp_block = BaseFNN(indim=indim, **one_mlp_setting)
                self.mlp_blcoks.append(one_mlp_block)
                feats_dim = one_mlp_setting['dim']
        else:
            feats_dim = bottle_neck if bottle_neck else cnov_outdim
        self.logit = nn.Linear(feats_dim, num_class)
        self.sig = nn.Sigmoid()
        self.num_class = num_class
        self.ptarget = ptarget
        self.pfeats = pfeats
        self.homo_pos = homo_pos
        self.mt = mt
        if loss == 'xent':
            self.crit_loss = nn.CrossEntropyLoss()
        else:
            self.crit_loss = nn.BCEWithLogitsLoss()
        #self.xent_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.homo_weight = homo_weight
        assert(task in ['base', 'mixup', 'homomixup', 'homo', 'homomixnonetarget', 'augment', 'hard_mixup','augmixup'])
        self.task = task
    
    def forward_module(self, input, module):
        acts = []
        for i, sub_m in enumerate(module):
            if not isinstance(sub_m, nn.Identity):
                input, act = sub_m(input)
            else:
                input = sub_m(input)
                act = torch.empty(1,1)
            acts.append(act.clone())
        return input, acts

    def compute_homo(self, c_act, i_act, m_act, ratio):
        if not isinstance(c_act, list):
            c_act = [c_act]
            i_act = [i_act]
            m_act = [m_act]

        for i, cact in enumerate(c_act):
            #remix = cact + i_act[i] * ratio
            b = cact.size(0)
            if cact.dim() == 2:
                r = ratio.view(b,1)
            if cact.dim() == 4:
                r = ratio.view(b,1,1,1)

            remix = r * cact + (1-r)*i_act[i]
            one_homo_loss =  self.mse_loss(remix, m_act[i])

            if i == 0:
                homo_loss = one_homo_loss
            else:
                homo_loss = homo_loss + one_homo_loss
        return homo_loss
    
    def forward_acoustice(self, feats, return_emb=False):
        feats, cnn_act = self.forward_module(feats.unsqueeze(1), self.conv_blocks)
        feats = self.pooling(feats)
        feats = self.pooling_act(feats)
        b = feats.size(0)
        pooling_act = [feats.clone()]
        feats = feats.view(b, -1)
        feats = self.bottle_neck(feats)
        feats, mlp_act = self.forward_module(feats, self.mlp_blcoks)
        if return_emb:
            emb = feats.clone()
        feats = self.logit(feats)
        if self.homo_pos == None:
            acts = cnn_act + pooling_act + mlp_act
        elif self.homo_pos == 'cnn':
            acts = cnn_act
        elif self.homo_pos == 'pooling':
            acts = pooling_act
        elif self.homo_pos == 'mlp':
            acts = mlp_act
        else:
            raise NotImplementedError("homo pos = [cnn, pooling, mlp]")
        if return_emb:
            return feats, acts, emb
        else:
            return feats, acts
    
    def forward(self, inputs):
        detail_loss = {}

        if self.task in ['base', 'augment']:
            speech, target = inputs
            target = F.one_hot(target.view(-1), num_classes=self.num_class).to(torch.float)
            logit, _ = self.forward_acoustice(speech)
            #loss = self.bce_loss(logit1, target1)
            loss = self.crit_loss(logit, target)
            detail_loss.update({'det_loss': loss.clone()})
        elif self.task in ['mixup', 'hard_mixup', 'augmixup']:
            mixspeech, speech, targets, ratios = inputs
            targets = F.one_hot(targets, num_classes=self.num_class).to(torch.float)
            # forward mix speech
            if self.task in ['hard_mixup','augmixup']:
                soft=False
            else:
                soft=True
            n_target = targets.size(1)
            mix_target = make_mix_target(targets, ratios[:,0:n_target], soft=soft)
            mix_logit, _ = self.forward_acoustice(mixspeech)
            mix_loss = self.crit_loss(mix_logit, mix_target)
            loss = mix_loss
            detail_loss.update({"mix_det_loss": mix_loss.clone()})

            # forward clean speech
            if self.mt:
                b, n, t, d = speech.size()
                speech = speech.transpose(0,1).contiguous() # b, n, d => n, b, d
                for i, one_clean in enumerate(speech): # one_clean: b, 
                    if i>= 1:
                        continue
                    one_target = targets[:,i]
                    one_clean_logit, _ = self.forward_acoustice(one_clean)
                    one_loss = self.crit_loss(one_clean_logit, one_target)
                    if i == 0:
                        clean_det_loss = one_loss
                    else:
                        clean_det_loss  = clean_det_loss + one_loss
                loss = loss + clean_det_loss
                detail_loss.update({
                    "clean_det_loss": clean_det_loss.clone(),
                })

            detail_loss.update({'total_loss': loss})
        #elif self.task == "xxxx" :
        #    speech, target1, target2, ratio1, ratio2 = inputs
        #    b = speech.size(0)
        #    target1 = F.one_hot(target1.view(-1), num_classes=self.num_class).to(torch.float)
        #    target2 = F.one_hot(target2.view(-1), num_classes=self.num_class).to(torch.float)
        #    target = ratio1.view(b,-1) * target1 + ratio2.view(b,-1) * target2
        #    logit, _ = self.forward_acoustice(speech)
        #    loss = self.crit_loss(logit, target)
        #    detail_loss.update({'det_loss': loss.clone()})
        #elif self.task in ['hard_mixup','augmixup']:
        #    speech, target1, target2 = inputs
        #    target1 = F.one_hot(target1.view(-1), num_classes=self.num_class).to(torch.float)
        #    target2 = F.one_hot(target2.view(-1), num_classes=self.num_class).to(torch.float)
        #    target = target1 + target2
        #    target = torch.where(target >= 1, 1, 0).to(torch.float)
        #    logit, _ = self.forward_acoustice(speech)
        #    loss = self.crit_loss(logit, target)
        #    detail_loss.update({'det_loss': loss.clone()})
        #elif self.task == 'homo':
        #    mix_logit, mix_acts = self.forward_acoustice(mix_speech)
        #    logit1, logit_act1 = self.forward_acoustice(speech1)
        #    _, logit_act2 = self.forward_acoustice(speech2)
        #    if self.pfeats == 'clean':
        #        #loss = self.bce_loss(logit1, target1)
        #        loss = self.crit_loss(logit1, target1)
        #    else:
        #        #loss = self.bce_loss(mix_logit, target1)
        #        loss = self.crit_loss(logit1, target1)
        #    detail_loss.update({'det_loss': loss.clone()})
        #    homo_loss = self.compute_homo(c_act=logit_act1, i_act=logit_act2, m_act=mix_acts, ratio=ratio)
        #    homo_loss = self.homo_weight * homo_loss
        #    loss = loss + homo_loss
        #    detail_loss.update({"homo_loss": homo_loss.clone()})
        #elif self.task == 'homomixup':
        #    mix_logit, mix_acts = self.forward_acoustice(mix_speech)
        #    _, logit_act1 = self.forward_acoustice(speech1)
        #    _, logit_act2 = self.forward_acoustice(speech2)
        #    #loss = self.bce_loss(mix_logit, mix_target)
        #    loss = self.crit_loss(mix_logit, mix_target)
        #    detail_loss.update({'det_loss': loss.clone()})
        #    homo_loss = self.compute_homo(c_act=logit_act1, i_act=logit_act2, m_act=mix_acts, ratio=ratio)
        #    loss = loss + homo_loss
        #    detail_loss.update({"homo_loss": homo_loss.clone()})
        #elif self.task == 'homomixnonetarget':
        #    mix_logit, _ = self.forward_acoustice(mix_speech)
        #    _, logit_act1 = self.forward_acoustice(speech1)
        #    _, logit_act2 = self.forward_acoustice(speech2)
        #    #loss = self.bce_loss(mix_logit, mix_target)
        #    loss = self.crit_loss(mix_logit, mix_target)
        #    detail_loss.update({'det_loss': loss.clone()})
        #    _, mix_acts = self.forward_acoustice(mix_speech2)
        #    homo_loss = self.compute_homo(c_act=logit_act1, i_act=logit_act2, m_act=mix_acts, ratio=ratio2)
        #    loss = loss + homo_loss
        #    detail_loss.update({"homo_loss": homo_loss.clone()})
        else:
            raise NotImplementedError("task should be [base, mixup, homo, homomixup]")
        return loss, detail_loss

    @torch.no_grad()
    def evaluate(self, input):
        speech, target = input
        #target1 = F.one_hot(target.view(-1), num_classes=self.num_class).to(torch.float)
        b = speech.size(0)
        logit, _, emb = self.forward_acoustice(speech, return_emb=True)
        #loss = self.crit_loss(logit, target1)
        #return self.sig(logit), target, loss
        #return logit.softmax(dim=-1), target, emb
        return self.sig(logit), target, emb
        #return logit.softmax(dim=-1), target, emb


        

