import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import model.NetModules as NM


def subsequent_mask( size, device):
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask

def pad_list(xs, pad_value):
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res

def add_sos_eos(ys_pad, sos, eos, ignore_id):
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)

att_dict = {
    'MultiHeadCrossAtt': NM.MultiHeadCrossAtt, 
    'MultiHeadAtt': NM.MultiHeadAtt
}

class TransformerKWSPhoneUnet(nn.Module):
    def __init__(
        self,
        audio_net_config,
        kw_net_config,
        num_audio_block=8,
        num_kw_block=4,
        sok=1,
        eok=1,
        batch_padding_idx=-1,
        loss_weight=[0.3,0.6,0.1],
        **kwargs,
    ):
        super(TransformerKWSPhoneUnet, self).__init__()
        self.sok = sok
        self.eok = eok
        self.batch_padding_idx = batch_padding_idx
        # detach_config 
        # TODO: looks stupid ....   >_<
        # audio config
        au_input_trans_config = audio_net_config['input_trans']
        au_transformer_config = audio_net_config['transformer_config']
        au_self_att = att_dict[au_transformer_config['self_att']]
        au_self_att_cofing = au_transformer_config['self_att_config']
        au_cross_att = att_dict[au_transformer_config['cross_att']]
        au_cross_att_config = au_transformer_config['corss_att_config']
        au_feed_forward_config = au_transformer_config['feed_forward_config']
        au_hidden_dim = au_transformer_config['size']

        # vocab config
        kw_input_trans_config = kw_net_config['input_trans']
        num_phn_token = kw_net_config['num_phn_token']
        kw_transformer_config = kw_net_config['transformer_config']
        kw_self_att = att_dict[kw_transformer_config['self_att']]
        kw_self_att_cofing = kw_transformer_config['self_att_config']
        kw_feed_forward_config = kw_transformer_config['feed_forward_config']
        kw_hidden_dim = kw_transformer_config['size']

        # audio net
        self.au_conv = nn.Sequential(
            torch.nn.Conv2d(1, au_hidden_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(au_hidden_dim, au_hidden_dim, 3, 2),
            torch.nn.ReLU(),
        )
        self.au_conv_trans = nn.Linear(au_hidden_dim * (((40 - 1) // 2 - 1) // 2), au_hidden_dim)

        self.au_trans = NM.FNNBlock(
            **au_input_trans_config
        )
        self.au_pos_emb = NM.PositionalEncoding(au_hidden_dim)
        assert (num_audio_block % 2 == 0)

        self.au_transformer_low = nn.ModuleList([
            NM.TransformerLayer(
                size=au_hidden_dim,
                self_att=au_self_att(**au_self_att_cofing),
                feed_forward=NM.FNNBlock(**au_feed_forward_config),
            ) for _ in range(num_audio_block // 2 - 1)
        ])

        self.au_transformer_mid = nn.ModuleList([
            NM.TransformerLayer(
                size=au_hidden_dim,
                self_att=au_self_att(**au_self_att_cofing),
                feed_forward=NM.FNNBlock(**au_feed_forward_config),
            ) for idx in range(2)
        ])

        self.au_transformer_high = nn.ModuleList([
            NM.TransformerLayer(
                size=au_hidden_dim,
                self_att=au_self_att(**au_self_att_cofing),
                cross_att=au_cross_att(**au_cross_att_config),
                feed_forward=NM.FNNBlock(**au_feed_forward_config),
            ) for _ in range(num_audio_block // 2 - 1)
        ])
        self.skip_trans = nn.ModuleList([nn.Linear(au_hidden_dim, au_hidden_dim) for _ in range(num_audio_block // 2 - 1)])

        # kw net
        self.phn_emb = NM.WordEmbedding(
            num_tokens=num_phn_token, dim=kw_transformer_config['size']
        )
        self.kw_pos_emb = NM.PositionalEncoding(kw_hidden_dim)
        self.kw_trans = NM.FNNBlock(**kw_input_trans_config)
        self.kw_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=kw_hidden_dim,
                self_att=kw_self_att(**kw_self_att_cofing),
                feed_forward=NM.FNNBlock(**kw_feed_forward_config)
            ) for _ in range(num_kw_block)
        ])
        if kw_hidden_dim != au_hidden_dim:
            self.kw_au_link = nn.Linear(kw_hidden_dim, au_hidden_dim)
        else:
            self.kw_au_link = nn.Identity()

        self.location_cross_att = NM.MultiHeadCrossAtt(**au_cross_att_config)
        self.det_net = nn.Sequential(
            NM.FNNBlock(**au_feed_forward_config), nn.Linear(au_hidden_dim, 1), nn.Sigmoid()
        )

        # decoder net
        
        phn_ctc_conf = {
            'num_tokens': num_phn_token,
            'front_output_size': au_hidden_dim 
        }
        self.l1, self.l2, self.l3 = loss_weight
        self.det_crit = nn.BCELoss(reduction='mean')
        self.phn_asr_crit = NM.CTC(**phn_ctc_conf)

    def forward_transformer(
        self,
        transformer_module,
        input,
        mask=None,
        cross_embedding=None,
        analyse=False,
        print_mask=False
    ):
        if analyse:
            b = input.size(0)
            att_scores = {i:[] for i in range(b)}
            embeddings = {i:[] for i in range(b)}
        for i, tf_layer in enumerate(transformer_module):
            input, att_score = tf_layer(input, mask, cross_input=cross_embedding, print_mask=print_mask)
            if not analyse:
                continue
            for i, att in enumerate(att_score):
                att_scores[i].append(copy.deepcopy(att))
                embeddings[i].append(copy.deepcopy(input))
        if analyse:
            return input, (att_scores, embeddings)
        else:
            return input

    def make_align_loss(
        self, 
        masked_location_att, 
        kw_len, 
        sph_len,
        cross_mask,
        target
    ):
        masked_location_att = torch.sum(masked_location_att, dim=1)
        batch = masked_location_att.size(0)
        _, w, h = masked_location_att.size()
        cross_mask = cross_mask.transpose(-2,-1).squeeze(1)
        device = target.device
        w_matrix = torch.arange(1, w+1, device=device).unsqueeze(1).repeat(1, h)
        h_matrix = torch.arange(1, h+1, device=device).unsqueeze(0).repeat(w, 1)
        w_matrix = w_matrix.unsqueeze(0).repeat(batch,1,1) / kw_len[:,None,None]
        h_matrix = h_matrix.unsqueeze(0).repeat(batch,1,1) / sph_len[:,None,None]
        gau_mask = 1 - torch.exp(-(w_matrix-h_matrix)**2/2)
        gau_mask = gau_mask.masked_fill(cross_mask, 0.0)
        gau_loss = torch.mean(torch.sum(gau_mask * masked_location_att, dim=1), dim=1)
        aux_target1 = torch.where(target==1, 0, 1)
        aux_target2 = torch.where(target==1, 1, -1)
        gau_loss = aux_target1 + aux_target2*gau_loss
        return gau_loss 
    
    def forward_unet(self, input, mask=None, cross_embedding=None):
        res = {}
        for i, tf_layer in enumerate(self.au_transformer_low):
            input, att_score = tf_layer(input, mask, cross_input=None)
            res[i] = input
        
        for i, tf_layer in enumerate(self.au_transformer_mid):
            input, att_score = tf_layer(input, mask, cross_input=None)
        
        # revers res
        num_res = len(res)
        r_res = {i: res[num_res - i - 1] for i in range(num_res)}
        
        for i, tf_layer in enumerate(self.au_transformer_high):
            input = self.skip_trans[i](input + r_res[i])
            input, att_score = tf_layer(input, mask, cross_input=cross_embedding)
        return input
    

    def predict_kw_mask(self, asr_hyp):
        b, t, num_phone = asr_hyp.size()
        start_hyp = torch.argmax(
            asr_hyp[:,:,self.sok], dim=-1
        )
        end_hyp = torch.argmax(
            asr_hyp[:,:,self.eok], dim=-1
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

        sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, bpe_label, bpe_label_len, target = input_data
        b,t,d = sph_input.size()
        sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)

        # embedding
        sph_emb = self.au_conv(sph_input.unsqueeze(1))
        b, c, t, d = sph_emb.size()
        sph_emb = self.au_conv_trans(sph_emb.transpose(1,2).contiguous().view(b, t, c * d))
        sph_emb = self.au_trans(sph_emb)
        kw_emb = self.phn_emb(kw_label.to(torch.long))
        kw_emb = self.kw_trans(kw_emb)

        # add position embedding
        sph_emb = self.au_pos_emb(sph_emb)
        kw_emb = self.kw_pos_emb(kw_emb)

        kw_emb = self.forward_transformer(
            self.kw_transformer,
            kw_emb,
            mask=kw_mask,
        )
        sph_emb = self.forward_unet(sph_emb, mask=sph_mask, cross_embedding=(kw_emb, kw_emb, cross_mask))

        # asr loss
        phn_ctc_loss, phn_asr_hyp = self.phn_asr_crit(
            sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        )

        # pos loss
        kw_pos_mask = self.predict_kw_mask(phn_asr_hyp.transpose(0,1))
        cross_context, cross_att = self.location_cross_att(
            kw_emb, sph_emb, sph_emb, mask=cross_mask.transpose(-2,-1), aux_score=kw_pos_mask
        )
        align_loss = self.make_align_loss(cross_att, kw_len, sph_len, cross_mask, target)
        align_loss = torch.mean(align_loss)

        # detection loss
        det_result = self.det_net(cross_context[:,0,:])
        det_loss = self.det_crit(det_result, target.to(torch.float32))

        # decoder output 
        total_loss = (0.3 * phn_ctc_loss) + (0.6 * det_loss) + (0.1 * align_loss)
        detail_loss = {
            'phn_ctc_loss': phn_ctc_loss.clone().detach(),
            'det_loss': det_loss.clone().detach(),
            'align_loss': align_loss.clone().detach()
        }
        return total_loss, detail_loss
    

    @torch.no_grad()
    def evaluate(self, input_data):
        #sph_input, phn_label, kw_label, sph_len, phn_len, kw_len, target = input_data 
        sph_input, sph_len, phn_label, phn_len, kw_label, kw_len = input_data
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)
        #print (sph_input.size(), phn_label.size(), kw_label.size(), sph_len.size(), phn_len.size(), kw_len.size(), target.size())
        # embedding
        sph_emb = self.au_trans(sph_input)
        phn_emb = self.word_emb(kw_label.to(torch.long))
        phn_emb = self.kw_trans(phn_emb)

        # add position embedding
        sph_emb = self.au_pos_emb(sph_emb)
        phn_emb = self.kw_pos_emb(phn_emb)

        phn_emb = self.forward_transformer(
            self.kw_transformer,
            phn_emb,
            mask=kw_mask,
        )
        sph_emb, att_score = self.forward_transformer(
            self.au_transformer,
            sph_emb,
            mask=sph_mask,
            cross_embedding=(
                phn_emb, phn_emb, cross_mask
            ),
            analyse=True
        )
        asr_loss, asr_hyp = self.asr_crit(
            sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        )
        kw_pos_mask = self.predict_kw_mask(asr_hyp.transpose(0,1))
        return asr_hyp.transpose(0,1), sph_len, phn_label, att_score
