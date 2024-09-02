import math
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import model.NetModules as NM
import torchaudio

from torch.nn.utils import weight_norm

att_dict = {
    'MultiHeadCrossAtt': NM.MultiHeadCrossAtt, 
    'MultiHeadAtt': NM.MultiHeadAtt
}

#import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Vocoder(nn.Module):
    def __init__(self, input_size, ngf, n_residual_layers):
        super().__init__()
        ratios = [4,4,5,2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]
        self.model1 = nn.Sequential(*model)

        model = [] 
        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2
        self.model2 = nn.Sequential(*model)
        model = []
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model3 = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        return x


class ConformerHintSSVocoder(nn.Module):
    def __init__(
        self,
        audio_net_config,
        vocab_net_config,
        predict_token=None,
        num_audio_block=8,
        num_vocab_block=6,
        loss_weight=[0.3,0.6,0.1],
    ):
        super(ConformerHintSSVocoder, self).__init__()

        # detach_config 
        # TODO: looks stupid ....   >_<
        # audio config
        au_input_trans_config = audio_net_config['input_trans']
        au_conformer_config = audio_net_config['transformer_config']
        au_self_att = att_dict[au_conformer_config['self_att']]
        au_self_att_cofing = au_conformer_config['self_att_config']
        au_cross_att = att_dict[au_conformer_config['cross_att']]
        au_cross_att_config = au_conformer_config['corss_att_config']
        au_feed_forward_config = au_conformer_config['feed_forward_config']
        au_conv_config = au_conformer_config['conv_config']
        au_hidden_dim = au_conformer_config['size']

        # vocab config
        vocab_input_trans_config = vocab_net_config['input_trans']
        num_token = vocab_net_config['num_token']
        vocab_transformer_config = vocab_net_config['transformer_config']
        vocab_self_att = att_dict[vocab_transformer_config['self_att']]
        vocab_self_att_cofing = vocab_transformer_config['self_att_config']
        vocab_feed_forward_config = vocab_transformer_config['feed_forward_config']
        vocab_hidden_dim = vocab_transformer_config['size']

        if predict_token != None:
            self.kw_stoken, self.kw_etoken = predict_token 
        
        # vocab net
        self.vocab_emb = NM.WordEmbedding(
            num_tokens=num_token,dim=vocab_transformer_config['size']
        )
        self.vocab_pos_emb = NM.PositionalEncoding(vocab_hidden_dim)
        self.vocab_trans = NM.FNNBlock(**vocab_input_trans_config)
        self.vocab_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=vocab_hidden_dim,
                self_att=vocab_self_att(**vocab_self_att_cofing),
                feed_forward=NM.FNNBlock(**vocab_feed_forward_config)
            ) for _ in range(num_vocab_block)
        ])

        # am embedding net 
        ## input is filter bank
        ## output is speech embedding(SPE) SPE will feed forward to asr net and separation net
        self.au_trans = NM.FNNBlock(
            **au_input_trans_config
        )
        self.au_pos_emb = NM.PositionalEncoding(au_hidden_dim)
        self.au_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=au_hidden_dim,
                self_att=au_self_att(**au_self_att_cofing),
                cross_att=au_cross_att(**au_cross_att_config),
                feed_forward=NM.FNNBlock(**au_feed_forward_config),
                conv_layer=NM.DepthWiseConv(**au_conv_config)
            ) for _ in range(num_audio_block-4)
        ])

        # asr network: 2 Conv layer to do down sampling which is fancy to CTC Loss
        ## input is SPE speech embedding
        ## output is asr hyp
        asr_subsample_list = []
        self.t_reduc_factor = [(3,2), (3,2)] # kernal size along time axes
        d_reduc_factor = [(7,4), (7,4)] # kernal size along embedding dimension
        channels = [[1,32], [32, 64]]
        output_dim = au_hidden_dim
        for i in range(2):
            ic, oc = channels[i]
            one_conv = NM.BaseConv(ic, oc, nn.Conv2d, kernel_size=(3,7), stride=(2,4))
            output_dim = NM.BaseConv.compute_dim_redecution(output_dim, d_reduc_factor[i][0], d_reduc_factor[i][1], 0, 1)
            asr_subsample_list.append(one_conv)
        output_dim = 64 * output_dim
        self.asr_net = nn.Sequential(*asr_subsample_list)
        
        # separation net input
        ## input is SPE (speech embedding)
        ## output is separated filter bank feature
        self.sep_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=au_hidden_dim,
                self_att=au_self_att(**au_self_att_cofing),
                feed_forward=NM.FNNBlock(**au_feed_forward_config),
                conv_layer=NM.DepthWiseConv(**au_conv_config)
            ) for _ in range(4)
        ])
        spec_res_net = [
            nn.Linear(au_hidden_dim, 80), nn.ReLU()
        ]
        self.spec_res_net = nn.Sequential(*spec_res_net)

        # Vocoder 
        ## input is separated filter bank
        ## output is waveform
        self.vocoder = Vocoder(80, 32, 2) 

        ctc_conf = {
           'num_tokens': num_token,
           'front_output_size': output_dim 
        }
        self.asr_crit = NM.CTC(**ctc_conf)
        self.spec_crit = nn.L1Loss()
        self.l1_crit = nn.L1Loss()
        self.sdr_crit = NM.sisdr_loss

    def forward_transformer(
        self,
        transformer_module,
        input,
        mask=None,
        cross_embedding=None,
        analyse=False
    ):
        if analyse:
            b = input.size(0)
            att_scores = {i:[] for i in range(b) }
            embeddings = {i:[] for i in range(b)}
        for i, tf_layer in enumerate(transformer_module):
            input, att_score = tf_layer(input, mask, cross_input=cross_embedding)
            if not analyse:
                continue
            for i, att in enumerate(att_score):
                att_scores[i].append(copy.deepcopy(att))
                embeddings[i].append(copy.deepcopy(input))
        if analyse:
            return input, (att_scores, embeddings)
        else:
            return input
    
    def predict_kw_mask(self, asr_hyp):
        b, t, num_phone = asr_hyp.size()
        start_hyp = torch.argmax(
            asr_hyp[:,:,self.kw_stoken], dim=-1
        )
        end_hyp = torch.argmax(
            asr_hyp[:,:,self.kw_etoken], dim=-1
        )
        for i, (s, e)  in enumerate(zip(start_hyp, end_hyp)):
            s = s.item()
            e = e.item()
            if s >= e:
                one_mask = torch.ones(t, device=asr_hyp.device)
            else:
                head = list(map(
                    lambda x: math.exp(-(x-s)**2/0.5**2), 
                    [i for i in range(0, s)]
                ))
                mid = [1 for x in range(s, e)]
                tail = list(map(
                    lambda x: math.exp(-(x-e)**2/0.5**2),
                    [i for i in range(e, t)]
                ))
                one_mask = head + mid + tail
                one_mask = torch.tensor(one_mask, device=asr_hyp.device)
            if i == 0:
                mask = one_mask.unsqueeze(0)
            else:
                mask = torch.cat([mask, one_mask.unsqueeze(0)], dim=0)
        mask = torch.where(mask<1e-2, 1e-9, mask)
        return mask
    

    def forward(self, input_data):
        sph_input, sph_len, clean, clean_len, phn_label, phn_len, kw_label, kw_len, raw_wav = input_data
        b,t,d = sph_input.size()
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)
        loss_mask = ~NM.make_mask(sph_len)
        # embedding
        sph_emb = self.au_trans(sph_input)
        #sph_emb = self.au_trans(clean)
        phn_emb = self.vocab_emb(kw_label.to(torch.long))
        phn_emb = self.vocab_trans(phn_emb)

        # add position embedding
        sph_emb = self.au_pos_emb(sph_emb)
        phn_emb = self.vocab_pos_emb(phn_emb)

        phn_emb = self.forward_transformer(
            self.vocab_transformer,
            phn_emb,
            mask=kw_mask,
        )
        sph_emb = self.forward_transformer(
            self.au_transformer,
            sph_emb,
            mask=sph_mask,
            cross_embedding=(
                phn_emb, phn_emb, cross_mask
            )
        )
        asr_emb = self.asr_net(sph_emb.unsqueeze(1))
        for i in range(2):
            sph_len = NM.BaseConv.compute_dim_redecution(
                sph_len, self.t_reduc_factor[i][0], self.t_reduc_factor[i][1], 0, 1
        )
        b,c,t,d = asr_emb.size()
        asr_emb = asr_emb.transpose(1,2).contiguous().view(b,t,c*d)
        asr_loss, asr_hyp = self.asr_crit(
             asr_emb, phn_label, sph_len, phn_len, return_hyp=True
        )
        sph_emb = self.forward_transformer(
            self.sep_transformer,
            sph_emb,
            mask=sph_mask
        )
        
        b,t,d = sph_emb.size()
        loss_mask = loss_mask.eq(0)
        sph_rec = self.spec_res_net(sph_emb.view(b*t, d))
        sph_rec = sph_rec.view(b,t,-1)

        spec_loss = self.spec_crit(sph_rec, clean)
        spec_loss = spec_loss.mean()

        waveform = self.vocoder(sph_rec.transpose(1,2)) # b t d -> b d t
        waveform = waveform.squeeze(1)
        sdr_loss = self.sdr_crit(waveform, raw_wav)
        l1_loss = self.l1_crit(waveform, raw_wav)
        total_loss = spec_loss + asr_loss + sdr_loss + l1_loss

        detail_loss = {
            'spec_loss': spec_loss.clone().detach(),
            'asr_loss': asr_loss.clone().detach(),
            'sdr_loss': sdr_loss.clone().detach(),
            'l1_loss': l1_loss.clone().detach(),
        }
        return total_loss, detail_loss
    
    def generate_mask(self, sequence_length, batch_size, p=0.5):
        mask = torch.tensor(torch.bernoulli(torch.ones(batch_size, sequence_length) * p), dtype=torch.float32)
        return mask

    @torch.no_grad()
    def evaluate(self, input_data):
        sph_input, sph_len, phn_label, phn_len, kw_label, kw_len = input_data
        b,t,d = sph_input.size()
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)
        loss_mask = ~NM.make_mask(sph_len)
        # embedding
        sph_emb = self.au_trans(sph_input)
        #sph_emb = self.au_trans(clean)
        phn_emb = self.vocab_emb(kw_label.to(torch.long))
        phn_emb = self.vocab_trans(phn_emb)

        # add position embedding
        sph_emb = self.au_pos_emb(sph_emb)
        phn_emb = self.vocab_pos_emb(phn_emb)

        phn_emb = self.forward_transformer(
            self.vocab_transformer,
            phn_emb,
            mask=kw_mask,
        )
        sph_emb = self.forward_transformer(
            self.au_transformer,
            sph_emb,
            mask=sph_mask,
            cross_embedding=(
                phn_emb, phn_emb, cross_mask
            )
        )
        asr_emb = self.asr_subsample(sph_emb.unsqueeze(1))
        for i in range(2):
            sph_len = NM.BaseConv.compute_dim_redecution(
                sph_len, self.t_reduc_factor[i][0], self.t_reduc_factor[i][1], 0, 1
        )
        b,c,t,d = asr_emb.size()
        asr_emb = asr_emb.transpose(1,2).contiguous().view(b,t,c*d)
        asr_loss, asr_hyp = self.asr_crit(
             asr_emb, phn_label, sph_len, phn_len, return_hyp=True
        )
        print (asr_loss)
        sph_emb = self.forward_transformer(
            self.sep_transformer,
            sph_emb,
            mask=sph_mask
        )
        b,t,d = sph_emb.size()
        loss_mask = loss_mask.eq(0)
        sph_rec = self.spec_res_net(sph_emb.view(b*t, d))
        sph_rec = sph_rec.view(b,t,-1)

        return asr_hyp.transpose(0,1), sph_rec, phn_label, sph_len
