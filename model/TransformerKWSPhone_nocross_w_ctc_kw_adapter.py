import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import model.NetModules as NM
import model.TransformerKWSPhone_nocross_w_ctc as TransformerKWSPhone_nocross_w_ctc


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

class TransformerKWSPhone_nocross_w_ctc_kw_adapter(TransformerKWSPhone_nocross_w_ctc.TransformerKWSPhone_nocross_w_ctc):
        
    def __init__(
        self,
        audio_net_config,
        kw_net_config,
        au_kw_net_config,
        num_audio_self_block=4,
        num_kw_self_block=4,
        num_au_kw_concat_block=4,
        sok=1,
        eok=1,
        loss_weight=[0.3,0.7],
        **kwargs,
    ):
        super(TransformerKWSPhone_nocross_w_ctc_kw_adapter, self).__init__(
            audio_net_config=audio_net_config,
            kw_net_config=kw_net_config,
            au_kw_net_config=au_kw_net_config,
            num_audio_self_block=num_audio_self_block,
            num_kw_self_block=num_kw_self_block,
            num_au_kw_concat_block=num_au_kw_concat_block,
            sok=sok,
            eok=eok,
            loss_weight=loss_weight,
            **kwargs
        )
        kw_adapter_trans_config = kw_net_config['adapter_trans']
        kw_adapter_trans_dim = kw_adapter_trans_config['dim']
        self.kw_adapter_trans = nn.Linear(kw_adapter_trans_dim, kw_adapter_trans_dim)        


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
    

    def forward_au_transformer(self, input, mask=None):
        #res = {}
        for i, tf_layer in enumerate(self.au_transformer):
            input, att_score = tf_layer(input, mask, cross_input=None)
            #res[i] = input
        
        return input


    def forward_au_kw_transformer(self, input, mask=None):
        #res = {}
        for i, tf_layer in enumerate(self.au_kw_transformer):
            input, att_score = tf_layer(input, mask, cross_input=None)
            #res[i] = input
        
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

        sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, target = input_data
        b,t,d = sph_input.size()
        sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)

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
        kw_emb = self.kw_adapter_trans(kw_emb)
        sph_emb = self.forward_au_transformer(sph_emb, mask=sph_mask)
        sph_kw_emb = torch.cat([kw_emb, sph_emb], dim=1)
        sph_kw_mask = torch.cat([kw_mask, sph_mask], dim=-1)
        #sph_kw_emb = self.au_kw_pos_emb(sph_kw_emb)
        det_loss = 0
        detail_loss = {}
        for i, tf_layer in enumerate(self.au_kw_transformer):
            sph_kw_emb, _ = tf_layer(sph_kw_emb, sph_kw_mask, cross_input=None)

            # detection loss
            det_result_layer = self.det_net(sph_kw_emb[:,0,:])
            det_loss_layer = self.det_crit(det_result_layer, target.to(torch.float32))
            det_loss += det_loss_layer / len(self.au_kw_transformer)
            detail_loss['det_loss_layer_{}'.format(i)] = det_loss_layer.clone().detach()
        
        sph_emb = sph_kw_emb[:,kw_emb.size(1):,:]

        # asr loss
        phn_ctc_loss, phn_asr_hyp = self.phn_asr_crit(
            sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        )

        # decoder output 
        total_loss = (0.3 * phn_ctc_loss) + (0.7 * det_loss)
        detail_loss['phn_ctc_loss'] = phn_ctc_loss.clone().detach()
        # detail_loss = {
        #     'phn_ctc_loss': phn_ctc_loss.clone().detach(),
        #     'det_loss': det_loss.clone().detach()
        # }
        return total_loss, detail_loss
    

    @torch.no_grad()
    def evaluate(self, input_data):
        sph_input, sph_len, kw_label, kw_len = input_data
        b,t,d = sph_input.size()
        sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)

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
        kw_emb = self.kw_adapter_trans(kw_emb)
        sph_emb = self.forward_au_transformer(sph_emb, mask=sph_mask)
        sph_kw_emb = torch.cat([kw_emb, sph_emb], dim=1)
        sph_kw_mask = torch.cat([kw_mask, sph_mask], dim=-1)
        for i, tf_layer in enumerate(self.au_kw_transformer):
            sph_kw_emb, _ = tf_layer(sph_kw_emb, sph_kw_mask, cross_input=None)

            det_result = self.det_net(sph_kw_emb[:,0,:])
        
        sph_emb = sph_kw_emb[:,kw_emb.size(1):,:]

        phn_asr_hyp = self.phn_asr_crit.get_hyp(sph_emb)


        # decoder output 
        return det_result, phn_asr_hyp


    @torch.no_grad()
    def evaluate_sph_emb(self, input_data):
        sph_input, sph_len = input_data
        b,t,d = sph_input.size()
        sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)

        # embedding
        sph_emb = self.au_conv(sph_input.unsqueeze(1))
        b, c, t, d = sph_emb.size()
        sph_emb = self.au_conv_trans(sph_emb.transpose(1,2).contiguous().view(b, t, c * d))
        sph_emb = self.au_trans(sph_emb)

        # add position embedding
        sph_emb = self.au_pos_emb(sph_emb)

        sph_emb = self.forward_au_transformer(sph_emb, mask=sph_mask)

        return sph_emb, sph_mask
    
    @torch.no_grad()
    def evaluate_kw_emb(self, input_data):
        kw_label, kw_len = input_data
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)

        # embedding
        kw_emb = self.phn_emb(kw_label.to(torch.long))
        kw_emb = self.kw_trans(kw_emb)

        # add position embedding
        kw_emb = self.kw_pos_emb(kw_emb)

        kw_emb = self.forward_transformer(
            self.kw_transformer,
            kw_emb,
            mask=kw_mask,
        )
        kw_emb = self.kw_adapter_trans(kw_emb)

        return kw_emb, kw_mask
    
    @torch.no_grad()
    def evaluate_concat_attention(self, input_data):
        sph_emb, sph_mask, kw_emb, kw_mask = input_data

        sph_kw_emb = torch.cat([kw_emb, sph_emb], dim=1)
        sph_kw_mask = torch.cat([kw_mask, sph_mask], dim=-1)
        for i, tf_layer in enumerate(self.au_kw_transformer):
            sph_kw_emb, _ = tf_layer(sph_kw_emb, sph_kw_mask, cross_input=None)

            det_result = self.det_net(sph_kw_emb[:,0,:])

        sph_emb = sph_kw_emb[:,kw_emb.size(1):,:]

        phn_asr_hyp = self.phn_asr_crit.get_hyp(sph_emb)

        # decoder output 
        return det_result, phn_asr_hyp
