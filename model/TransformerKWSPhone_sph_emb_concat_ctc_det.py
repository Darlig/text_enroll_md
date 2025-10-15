import torch
import copy
import torch.nn as nn
import model.NetModules as NM


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

att_dict = {
    'MultiHeadCrossAtt': NM.MultiHeadCrossAtt, 
    'MultiHeadAtt': NM.MultiHeadAtt
}

class TransformerKWSPhone_sph_emb_concat_ctc_det(nn.Module):
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
        loss_weight={'ctc_loss': 0.3, 'det_loss': 0.7},
        **kwargs,
    ):
        super(TransformerKWSPhone_sph_emb_concat_ctc_det, self).__init__()
        self.sok = sok
        self.eok = eok
        # detach_config 
        # TODO: looks stupid ....   >_<
        # audio config
        au_transformer_config = audio_net_config['transformer_config']
        au_hidden_dim = au_transformer_config['size']

        # vocab config
        kw_input_trans_config = kw_net_config['input_trans']
        num_phn_token = kw_net_config['num_phn_token']
        kw_transformer_config = kw_net_config['transformer_config']
        kw_self_att = att_dict[kw_transformer_config['self_att']]
        kw_self_att_cofing = kw_transformer_config['self_att_config']
        kw_feed_forward_config = kw_transformer_config['feed_forward_config']
        kw_hidden_dim = kw_transformer_config['size']

        au_kw_transformer_config = au_kw_net_config['transformer_config']
        au_kw_self_att = att_dict[au_kw_transformer_config['self_att']]
        au_kw_self_att_cofing = au_kw_transformer_config['self_att_config']
        au_kw_feed_forward_config = au_kw_transformer_config['feed_forward_config']
        au_kw_hidden_dim = au_kw_transformer_config['size']

        self.ctc_weight = loss_weight['ctc_loss']
        self.det_weight = loss_weight['det_loss']

        # audio net
        self.au_pos_emb = NM.PositionalEncoding(au_hidden_dim)

        # kw net
        self.phn_emb = NM.WordEmbedding(
            num_tokens=num_phn_token, dim=kw_transformer_config['size']
        )
        self.kw_trans = NM.FNNBlock(**kw_input_trans_config)
        self.kw_pos_emb = NM.PositionalEncoding(kw_hidden_dim)
        self.kw_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=kw_hidden_dim,
                self_att=kw_self_att(**kw_self_att_cofing),
                feed_forward=NM.FNNBlock(**kw_feed_forward_config)
            ) for _ in range(num_kw_self_block)
        ])
        if kw_hidden_dim != au_hidden_dim:
            self.kw_au_link = nn.Linear(kw_hidden_dim, au_hidden_dim)
        else:
            self.kw_au_link = nn.Identity()

        # au kw concat net
        self.au_kw_pos_emb = NM.PositionalEncoding(au_kw_hidden_dim)
        
        # segment embedding for distinguishing speech and keyword segments
        self.segment_embedding = nn.Embedding(2, au_kw_hidden_dim)  # 0 for speech, 1 for keyword

        # separate token embedding
        self.separate_token_embedding = nn.Embedding(1, au_kw_hidden_dim)

        self.au_kw_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=au_kw_hidden_dim,
                self_att=au_kw_self_att(**au_kw_self_att_cofing),
                feed_forward=NM.FNNBlock(**au_kw_feed_forward_config),
            ) for _ in range(num_au_kw_concat_block)
        ])

        # decoder net
        phn_ctc_conf = {
            'num_tokens': num_phn_token,
            'front_output_size': au_hidden_dim 
        }
        self.phn_asr_crit = NM.CTC(**phn_ctc_conf)

        # detection net
        self.det_net = nn.Sequential(
            NM.FNNBlock(**kw_feed_forward_config), nn.Linear(kw_hidden_dim, 1)
        )
        self.det_crit = nn.BCEWithLogitsLoss(reduction='none')


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
    

    def forward_au_transformer(self, input, mask=None):
        #res = {}
        for i, tf_layer in enumerate(self.au_transformer):
            input, att_score = tf_layer(input, mask, cross_input=None)
            #res[i] = input
        
        return input


    def forward(self, input_data):

        sph_emb, sph_len, phn_label, phn_len, kw_label, kw_len, target = input_data
        # if torch.any(sph_len < 3):
        #     print(f"Rank {dist.get_rank()}: Skipping batch with invalid sph_len={sph_len}")
        #     return None, None
        # b,t,d = sph_input.size()
        # sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        # sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)

        # keyword embedding
        kw_emb = self.phn_emb(kw_label.to(torch.long))
        kw_emb = self.kw_trans(kw_emb)

        # add position embedding
        kw_emb = self.kw_pos_emb(kw_emb)

        kw_emb = self.forward_transformer(
            self.kw_transformer,
            kw_emb,
            mask=kw_mask
        )

        # Add segment embeddings
        batch_size, sph_seq_len, hidden_dim = sph_emb.shape
        _, kw_seq_len, _ = kw_emb.shape
        
        # Create segment IDs: 0 for speech, 1 for keyword
        sph_segment_ids = torch.zeros(batch_size, sph_seq_len, dtype=torch.long, device=sph_emb.device)
        kw_segment_ids = torch.ones(batch_size, kw_seq_len, dtype=torch.long, device=kw_emb.device)
        
        # Get segment embeddings
        sph_segment_emb = self.segment_embedding(sph_segment_ids)
        kw_segment_emb = self.segment_embedding(kw_segment_ids)
        # Add segment embeddings to the original embeddings
        sph_emb = sph_emb + sph_segment_emb
        kw_emb = kw_emb + kw_segment_emb

        # Create separate token embedding
        sep_token = torch.zeros(batch_size, 1, dtype=torch.long, device=sph_emb.device)
        sep_token_emb = self.separate_token_embedding(sep_token)
        sep_token_mask = torch.zeros(batch_size, 1, 1, dtype=torch.bool, device=sph_emb.device)

        sph_sep_kw_emb = torch.cat([sph_emb, sep_token_emb, kw_emb], dim=1)
        sph_sep_kw_mask = torch.cat([sph_mask, sep_token_mask, kw_mask], dim=-1)
        sph_sep_kw_emb = self.forward_transformer(
            self.au_kw_transformer,
            sph_sep_kw_emb,
            mask=sph_sep_kw_mask
        )

        # asr loss
        sph_emb = sph_sep_kw_emb[:,0:sph_emb.size(1),:]
        phn_ctc_loss, phn_asr_hyp = self.phn_asr_crit(
            sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        )

        # detection loss
        det_logit = self.det_net(sph_sep_kw_emb[:,sph_emb.size(1)+1:,:])
        det_logit = det_logit[:,:,0]
        # 使用mask排除padding位置
        det_loss_mask = kw_mask.squeeze(1)
        det_loss = self.det_crit(det_logit, target.to(torch.float32))
        det_loss = (det_loss * det_loss_mask).sum() / det_loss_mask.sum()

        # total loss
        total_loss = (self.ctc_weight * phn_ctc_loss) + (self.det_weight * det_loss)
        detail_loss = {}
        detail_loss['phn_ctc_loss'] = phn_ctc_loss.clone().detach()
        detail_loss['det_loss'] = det_loss.clone().detach()

        return total_loss, detail_loss
    

    @torch.no_grad()
    def evaluate(self, input_data):
        sph_emb, sph_len, kw_label, kw_len = input_data
        # b,t,d = sph_emb.size()
        # sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        # sph_len = NM.BaseConv.compute_dim_redecution(sph_len, 3, 2, 0, 1)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)

        # embedding
        kw_emb = self.phn_emb(kw_label.to(torch.long))
        kw_emb = self.kw_trans(kw_emb)

        # add position embedding
        kw_emb = self.kw_pos_emb(kw_emb)

        kw_emb = self.forward_transformer(
            self.kw_transformer,
            kw_emb,
            mask=kw_mask
        )
        
        # Add segment embeddings
        batch_size, sph_seq_len, hidden_dim = sph_emb.shape
        _, kw_seq_len, _ = kw_emb.shape
        
        # Create segment IDs: 0 for keyword, 1 for speech
        sph_segment_ids = torch.zeros(batch_size, sph_seq_len, dtype=torch.long, device=sph_emb.device)
        kw_segment_ids = torch.ones(batch_size, kw_seq_len, dtype=torch.long, device=kw_emb.device)
        
        # Get segment embeddings
        sph_segment_emb = self.segment_embedding(sph_segment_ids)
        kw_segment_emb = self.segment_embedding(kw_segment_ids)
        
        # Add segment embeddings to the original embeddings
        sph_emb = sph_emb + sph_segment_emb
        kw_emb = kw_emb + kw_segment_emb

        # Create separate token embedding
        sep_token = torch.zeros(batch_size, 1, dtype=torch.long, device=sph_emb.device)
        sep_token_emb = self.separate_token_embedding(sep_token)
        sep_token_mask = torch.zeros(batch_size, 1, 1, dtype=torch.bool, device=sph_emb.device)

        sph_sep_kw_emb = torch.cat([sph_emb, sep_token_emb, kw_emb], dim=1)
        sph_sep_kw_mask = torch.cat([sph_mask, sep_token_mask, kw_mask], dim=-1)
        sph_sep_kw_emb = self.forward_transformer(
            self.au_kw_transformer,
            sph_sep_kw_emb,
            mask=sph_sep_kw_mask
        )

        sph_emb = sph_sep_kw_emb[:,0:sph_emb.size(1),:]
        phn_asr_hyp = self.phn_asr_crit.get_hyp(sph_emb)

        det_result = self.det_net(sph_sep_kw_emb[:,sph_emb.size(1)+1:,:])[:,:,0]
        det_result = torch.sigmoid(det_result)

        # decoder output 
        return det_result, phn_asr_hyp


    @torch.no_grad()
    def greedy_decode(self, phn_asr_hyp):
        return self.phn_asr_crit.ctc_greedy_decode(phn_asr_hyp)


    @torch.no_grad()
    def compute_gop(self, phn_asr_hyp, phn_label):
        phn_asr_hyp = phn_asr_hyp.squeeze(0)
        phn_label = phn_label.squeeze(0)
        ali_spans = self.phn_asr_crit.ctc_forced_align_viterbi(phn_asr_hyp, phn_label)
        gop_result = self.phn_asr_crit.gop_avg_ctc_max_norm(phn_asr_hyp, ali_spans, phn_label)
        return gop_result

