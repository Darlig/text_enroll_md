import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import model.NetModules as NM


def label_batch_padding(label1, label2, padding_value=-1):
    max_len = max(label1.size(1), label2.size(1))
    device = label1.device
    if label1.size(1) < max_len:
        pad_len = max_len - label1.size(1)
        pad_v = torch.ones((label1.size(0), pad_len), device=device)
        label1 = torch.cat([label1, pad_v], dim=-1)
    if label2.size(1) < max_len:
        pad_len = max_len - label2.size(1)
        pad_v = torch.ones((label2.size(0), pad_len), device=device)
        label2 = torch.cat([label2, pad_v], dim=-1)
    return label1, label2

def label_re_padding(label):
    device = label.device
    b = label.size(0)
    de_label = [one[one != -1] for one in label]
    max_len = max([len(one) for one in de_label])
    new_label = torch.zeros(b, max_len, device=label.device)
    new_length = torch.zeros(b, device=device)
    for i in range(b):
        new_label[i, :len(de_label[i])] = de_label[i]
        new_length[i] = len(de_label[i])
    return new_label.to(torch.long), new_length.to(torch.long)
    
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

class CohortASR(nn.Module):
    def __init__(
        self,
        audio_net_config,
        decoder_net_config,
        num_audio_block=8,
        num_kw_block=3,
        sos=1,
        eos=1,
        num_decoder_block=4,
        batch_padding_idx=-1,
        loss_weight=[0.0,0.6,0.1],
        **kwargs,
    ):
        super(CohortASR, self).__init__()
        self.sos = sos
        self.eos = eos
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
        au_conv_config = au_transformer_config['conv_config']

        # decoder config
        decoder_input_trans_config = decoder_net_config['input_trans']
        num_bpe_token = decoder_net_config['num_bpe_token']
        num_phn_token = decoder_net_config['num_phn_token']
        decoder_transformer_config = decoder_net_config['transformer_config']
        decoder_self_att = att_dict[decoder_transformer_config['self_att']]
        decoder_self_att_cofing = decoder_transformer_config['self_att_config']
        decoder_cross_att = att_dict[decoder_transformer_config['cross_att']]
        decoder_cross_att_config = decoder_transformer_config['corss_att_config']
        decoder_feed_forward_config = decoder_transformer_config['feed_forward_config']
        decoder_hidden_dim = decoder_transformer_config['size']

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
        self.au_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=au_hidden_dim,
                self_att=au_self_att(**au_self_att_cofing),
                feed_forward=NM.FNNBlock(**au_feed_forward_config),
                conv_layer=NM.DepthWiseConv(**au_conv_config)
            ) for _ in range(num_audio_block)
        ])

        # kw net
        self.bpe_emb = NM.WordEmbedding(
            num_tokens=num_bpe_token, dim=decoder_transformer_config['size']
        )
        
        # decoder net
        #self.decoder_emb = NM.WordEmbedding(num_tokens=num_token, dim=decoder_transformer_config['size'])
        self.decoder_trans = NM.FNNBlock(
            **decoder_input_trans_config
        )
        self.decoder_pos_emb = NM.PositionalEncoding(decoder_hidden_dim)
        self.decoder_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=decoder_hidden_dim,
                self_att=decoder_self_att(**decoder_self_att_cofing),
                cross_att=decoder_cross_att(**decoder_cross_att_config),
                feed_forward=NM.FNNBlock(**decoder_feed_forward_config),
            ) for _ in range(num_decoder_block)
        ])
        if au_hidden_dim != decoder_hidden_dim:
            self.enc_dec_link = nn.Linear(au_hidden_dim, decoder_hidden_dim)
        else:
            self.enc_dec_link = nn.Identity()
        self.decoder_proj = nn.Linear(decoder_hidden_dim, num_bpe_token)
        
        bpe_ctc_conf = {
            'num_tokens': num_bpe_token,
            'front_output_size': au_hidden_dim 
        }
        phn_ctc_conf = {
            'num_tokens': num_phn_token,
            'front_output_size': au_hidden_dim ,
            'reduce': False
        }
        self.l1, self.l2, self.l3 = loss_weight
        self.bpe_asr_crit = NM.CTC(**bpe_ctc_conf)
        self.phn_asr_crit = NM.CTC(**phn_ctc_conf)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.lsl = NM.LabelSmoothingLoss(size=num_bpe_token,padding_idx=-1, smoothing=0.1, normalize_length=False)

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

    def forward_decoder(
        self,
        enc_emb,
        enc_emb_mask,
        label,
        label_len
    ):
        label_in_pad, label_out_pad = add_sos_eos(label, self.sos, self.eos, self.batch_padding_idx)
        label_in_pad = label_in_pad.to(torch.long)
        label_out_pad = label_out_pad.to(torch.long)
        label_len  = label_len + 1
        label_mask = ~NM.make_mask(label_len).unsqueeze(1)
        subseq_mask = subsequent_mask(label_mask.size(-1), device=label_mask.device).unsqueeze(0)
        real_mask = label_mask & subseq_mask
        label_emb = self.bpe_emb(label_in_pad)
        label_emb = self.decoder_pos_emb(label_emb)
        label_emb = self.decoder_trans(label_emb)
        #print (real_mask.size(), enc_emb_mask.size())
        #cross_mask = ~NM.combine_mask(real_mask, enc_emb_mask.squeeze(1), 1)
        label_emb = self.forward_transformer(
            self.decoder_transformer,
            label_emb,
            mask=real_mask,
            cross_embedding=(
                enc_emb, enc_emb, enc_emb_mask
            ),
            print_mask=False
        )
        output = self.decoder_proj(label_emb)
        return output, label_out_pad
    
    def forward(self, input_data, anchor, spk1, spk2):
        #sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, target = input_data
        #mixspeech,mixspeech_len,phn_label,phn_label_len,keyword,keyword_len,bpe_label,bpe_label_len,target
        sph_input, sph_len, bpe_label1, bpe_label1_len, bpe_label2, bpe_label2_len, phn_label1, phn_label1_len, phn_label2, phn_label2_len = input_data
        b,t,d = sph_input.size()
        sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_len = sph_len + 1
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1) # sph embedding will conject with a anchor vector [n x 1 x d]

        # embedding
        sph_emb = self.au_conv(sph_input.unsqueeze(1))
        b, c, t, d = sph_emb.size()
        sph_emb = self.au_conv_trans(sph_emb.transpose(1,2).contiguous().view(b, t, c * d))
        sph_emb = self.au_trans(sph_emb)
        b, t, d = sph_emb.size()
        anchor = anchor.view(1, 1, -1)
        anchor = anchor.repeat(b, 1, 1)

        # anchor : b x 1 x d
        sph_emb = torch.cat([anchor, sph_emb], dim=1)

        # add position embedding
        sph_emb = self.au_pos_emb(sph_emb)
        sph_emb = self.forward_transformer(
            self.au_transformer,
            sph_emb,
            mask=sph_mask,
        )

        #phn_ctc_loss1 = self.phn_asr_crit(
        #    sph_emb, phn_label1, sph_len, phn_label1_len, return_hyp=False
        #)
        #phn_ctc_loss2 = self.phn_asr_crit(
        #    sph_emb, phn_label2, sph_len, phn_label2_len, return_hyp=False
        #)
        cos1 = self.cos(anchor, spk1)
        cos2 = self.cos(anchor, spk2)
        label1_mask = cos1 <= cos2
        label2_mask = cos2 < cos1
        #phn_ctc_loss1 = phn_ctc_loss1.masked_fill(label1_mask, 0.0)
        #phn_ctc_loss2 = phn_ctc_loss2.masked_fill(label2_mask, 0.0)
        #phn_ctc_loss = phn_ctc_loss1 + phn_ctc_loss2
        #phn_ctc_loss = phn_ctc_loss.sum() / sph_emb.size(0)
        bpe_label1, bpe_label2 = label_batch_padding(bpe_label1, bpe_label2)
        bpe_label = bpe_label1.masked_fill(label1_mask.unsqueeze(-1), 0).to(torch.long) + bpe_label2.masked_fill(label2_mask.unsqueeze(-1), 0).to(torch.long)
        bpe_label = torch.where(bpe_label==0, -1, bpe_label)
        bpe_label_len = bpe_label1_len.masked_fill(label1_mask, 0) + bpe_label2_len.masked_fill(label2_mask, 0)
        bpe_label, bpe_label_len = label_re_padding(bpe_label)
        bpe_ctc_loss = self.bpe_asr_crit(
            sph_emb, bpe_label, sph_len, bpe_label_len, return_hyp=False
        )
        # decoder output 
        decoder_output, label_out_pad = self.forward_decoder(sph_emb, sph_mask, bpe_label, bpe_label_len)
        att_loss = self.lsl(decoder_output, label_out_pad)
        #total_loss = (0.3 * bpe_ctc_loss) + (0.1 * phn_ctc_loss) + (0.6 * att_loss)
        total_loss = (0.3 * bpe_ctc_loss)  + (0.7 * att_loss)
        #total_loss = att_loss
        detail_loss = {
            'bpe_ctc_loss': bpe_ctc_loss,
            'att_loss': att_loss.clone().detach()
        }
        return total_loss, detail_loss
    

    @torch.no_grad()
    def evaluate(self, input_data):
        #sph_input, phn_label, kw_label, sph_len, phn_len, kw_len, target = input_data 
        sph_input, sph_len, phn_label, phn_len, kw_label, kw_len = input_data
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
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
            analyse=True
        )
        asr_loss, asr_hyp = self.asr_crit(
            sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        )
        kw_pos_mask = self.predict_kw_mask(asr_hyp.transpose(0,1))
        #cross_context, cross_att = self.location_cross_att(
        #    phn_emb, sph_emb, sph_emb, mask=cross_mask.transpose(-2,-1), aux_score=kw_pos_mask
        #)
        
       
        # compute asr result
        #asr_hyp = self.asr_crit.get_hyp(sph_emb)
        #asr_loss, asr_hyp = self.asr_crit(
        #    sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        #)
        # compute kws location
        #kw_pos_mask = self.predict_kw_mask(asr_hyp.transpose(0,1))
        #cross_context, cross_att = self.location_cross_att(
            #phn_emb, sph_emb, sph_emb, mask=cross_mask.transpose(-2,-1), aux_score=kw_pos_mask
        #)
        #return asr_hyp.transpose(0,1), sph_len, phn_label, att_score, cross_att, sph_emb
        #return asr_hyp.transpose(0,1), sph_len, phn_label, cross_context
        return asr_hyp.transpose(0,1), sph_len, phn_label, att_score
