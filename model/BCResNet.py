import torch
from torch import Tensor
from model.NetModules import make_mix_target
import torch.nn as nn
import torch.nn.functional as F


class SubSpectralNorm(nn.Module):
    def __init__(self, C, S, eps=1e-5):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = nn.BatchNorm2d(C*S)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)

        x = self.bn(x)

        return x.view(N, C, F, T)


class BroadcastedBlock(nn.Module):
    def __init__(
            self,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(BroadcastedBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation,
                                      stride=stride, bias=False)
        self.ssn1 = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.5)
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)

        return out


class TransitionBlock(nn.Module):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(TransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride,
                                      dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.5)
        self.swish = nn.SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # f2
        #############################
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.freq_dw_conv(out)
        out = self.ssn(out)
        #############################
        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        #############################

        out = auxilary + out
        out = self.relu(out)

        return out


class BCResNet(torch.nn.Module):
    def __init__(
        self,
        num_class=36,
        t=8,
        task='base',
        loss='xent',
        mt=False
    ):
        super(BCResNet, self).__init__()
        assert (isinstance(t, int))
        self.conv1 = nn.Conv2d(1, 16*t, 5, stride=(2, 1), padding=(2, 2))
        self.block1_1 = TransitionBlock(16*t, 8*t)
        self.block1_2 = BroadcastedBlock(8*t)

        self.block2_1 = TransitionBlock(8*t, 12*t, stride=(2, 1), dilation=(1, 2), temp_pad=(0, 2))
        self.block2_2 = BroadcastedBlock(12*t, dilation=(1, 2), temp_pad=(0, 2))

        self.block3_1 = TransitionBlock(12*t, 16*t, stride=(2, 1), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_2 = BroadcastedBlock(16*t, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_3 = BroadcastedBlock(16*t, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_4 = BroadcastedBlock(16*t, dilation=(1, 4), temp_pad=(0, 4))

        self.block4_1 = TransitionBlock(16*t, 20*t, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_2 = BroadcastedBlock(20*t, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_3 = BroadcastedBlock(20*t, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_4 = BroadcastedBlock(20*t, dilation=(1, 8), temp_pad=(0, 8))

        self.conv2 = nn.Conv2d(20*t, 20*t, 5, groups=20*t, padding=(0, 2))
        self.conv3 = nn.Conv2d(20*t, 32*t, 1, bias=False)
        self.conv4 = nn.Conv2d(32*t, 12, 1, bias=False)
        self.fc = nn.Linear(12*6, 36)
        self.num_class = num_class
        self.task = task
        self.mt = mt
        if loss == 'xent':
            self.crit_loss = nn.CrossEntropyLoss()
        else:
            self.crit_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs):
        # mixspeech,speech,augspeech,word_keyword,ratio,num_feats

        detail_loss = {}
        if self.task in ['base', 'augment']:
            speech, target = inputs
            target = F.one_hot(target.view(-1), num_classes=self.num_class).to(torch.float)
            logit = self.forward_net(speech)
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
            mix_logit = self.forward_net(mixspeech)
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
                    one_clean_logit = self.forward_net(one_clean)
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
        return loss, detail_loss

    def forward_net(self, x):
        x = x.transpose(1,2).contiguous()
        x = x.unsqueeze(1)
        #x, target = inputs
        #print('INPUT SHAPE:', x.shape)
        b = x.size(0)
        out = self.conv1(x)

        #print('BLOCK1 INPUT SHAPE:', out.shape)
        out = self.block1_1(out)
        out = self.block1_2(out)

        #print('BLOCK2 INPUT SHAPE:', out.shape)
        out = self.block2_1(out)
        out = self.block2_2(out)

        #print('BLOCK3 INPUT SHAPE:', out.shape)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

        #print('BLOCK4 INPUT SHAPE:', out.shape)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

        #print('Conv2 INPUT SHAPE:', out.shape)
        out = self.conv2(out)

        #print('Conv3 INPUT SHAPE:', out.shape)
        out = self.conv3(out)
        out = out.mean(-1, keepdim=True)

        #print('Conv4 INPUT SHAPE:', out.shape)
        out = self.conv4(out)
        out = self.fc(out.view(b,-1))
        #loss = self.crit_loss(out, target)
        #detail_loss = {'det_loss': loss}

        #print('OUTPUT SHAPE:', out.shape)
        return out

    def evaluate(self, inputs):
        x, target = inputs
        x = x.transpose(1,2).contiguous()
        x = x.unsqueeze(1)
        target1 = F.one_hot(target.view(-1), num_classes=self.num_class).to(torch.float)
        #print('INPUT SHAPE:', x.shape)
        b = x.size(0)
        out = self.conv1(x)

        #print('BLOCK1 INPUT SHAPE:', out.shape)
        out = self.block1_1(out)
        out = self.block1_2(out)

        #print('BLOCK2 INPUT SHAPE:', out.shape)
        out = self.block2_1(out)
        out = self.block2_2(out)

        #print('BLOCK3 INPUT SHAPE:', out.shape)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

        #print('BLOCK4 INPUT SHAPE:', out.shape)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

        #print('Conv2 INPUT SHAPE:', out.shape)
        out = self.conv2(out)

        #print('Conv3 INPUT SHAPE:', out.shape)
        out = self.conv3(out)
        out = out.mean(-1, keepdim=True)

        #print('Conv4 INPUT SHAPE:', out.shape)
        out = self.conv4(out)
        out = self.fc(out.view(b,-1))
        #loss = self.crit_loss(out, target1)

        #print('OUTPUT SHAPE:', out.shape)
        return out.softmax(dim=-1), target, out

