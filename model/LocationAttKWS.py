import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.NetModules as NM

att_dict = {
    'MultiHeadCrossAtt': NM.MultiHeadCrossAtt, 
    'MultiHeadAtt': NM.MultiHeadAtt
}


class LocationAttKWS(nn.Module):
    def __init__(
        self,
        audio_net_config,
        vocab_net_config,
        predict_token=None,
        num_audio_block=8,
        num_vocab_block=6,
        audio_rnn=False,
        loss_weight=[0.3,0.6,0.1],
    ):
        super(LocationAttKWS, self).__init__()

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
        vocab_input_trans_config = vocab_net_config['input_trans']
        num_token = vocab_net_config['num_token']
        vocab_transformer_config = vocab_net_config['transformer_config']
        vocab_self_att = att_dict[vocab_transformer_config['self_att']]
        vocab_self_att_cofing = vocab_transformer_config['self_att_config']
        vocab_feed_forward_config = vocab_transformer_config['feed_forward_config']
        vocab_hidden_dim = vocab_transformer_config['size']

        if predict_token != None:
            self.kw_stoken, self.kw_etoken = predict_token 
        
        # audio net
        self.audio_rnn = audio_rnn
        if audio_rnn:
            self.audio_rnn_net = nn.LSTM(
                input_size=au_hidden_dim,
                hidden_size=int(au_hidden_dim / 2),
                batch_first=True,
                bidirectional=True
            )
        else:
            self.audio_rnn_net = nn.Identity()
   
        self.au_trans = NM.FNNBlock(
            **au_input_trans_config
        )
        self.au_pos_emb = NM.PositionalEncoding(au_hidden_dim)
        self.au_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=au_hidden_dim,
                self_att=au_self_att(**au_self_att_cofing),
                cross_att=au_cross_att(**au_cross_att_config),
                feed_forward=NM.FNNBlock(**au_feed_forward_config)
            ) for _ in range(num_audio_block)
        ])

        # vocab net
        self.vocab_emb = NM.WordEmbedding(
            num_tokens=num_token,dim=vocab_transformer_config['size']
        )
        #self.vocab_emb = nn.Embedding(num_token, vocab_transformer_config['size'])
        self.vocab_pos_emb = NM.PositionalEncoding(vocab_hidden_dim)
        self.vocab_trans = NM.FNNBlock(**vocab_input_trans_config)
        self.vocab_transformer = nn.ModuleList([
            NM.TransformerLayer(
                size=vocab_hidden_dim,
                self_att=vocab_self_att(**vocab_self_att_cofing),
                feed_forward=NM.FNNBlock(**vocab_feed_forward_config)
            ) for _ in range(num_vocab_block)
        ])

        ## final cross att
        self.location_cross_att = NM.MultiHeadCrossAtt(**au_cross_att_config)
        
        det_config = {
            'dim': au_hidden_dim,
            'norm': 'LayerNorm',
            'act': 'ReLU'
        }

        det_net = [
            NM.FNNBlock(**det_config), nn.Linear(au_hidden_dim, 1), nn.Sigmoid()
        ]
        self.det_net = nn.Sequential(*det_net)

        ctc_conf = {
            'num_tokens': num_token,
            'front_output_size': au_hidden_dim
        }
        self.l1, self.l2, self.l3 = loss_weight
        self.asr_crit = NM.CTC(**ctc_conf)
        self.det_crit = nn.BCELoss(reduction='mean')

    def forward_transformer(
        self,
        transformer_module,
        input,
        mask=None,
        cross_embedding=None,
        analyse=False
    ):
        for i, tf_layer in enumerate(transformer_module):
            input, att_score = tf_layer(input, mask, cross_input=cross_embedding)
        if analyse:
            return input, att_score
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

    def forward(self, input_data):
        #sph_input, phn_label, kw_label, sph_len, phn_len, kw_len, target = input_data 
        sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, target = input_data
        b,t,d = sph_input.size()
        #mixspeech, mixspeech_len,label,label_len,keyword,keyword_len,target
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)
        # embedding
        sph_emb = self.au_trans(sph_input)
        #sph_emb = self.au_trans(sph_input)
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
        if self.audio_rnn:
            sorted_seq_lengths, index = torch.sort(sph_len.cpu().to(torch.int64), descending=True)
            sph_emb = sph_emb[index]
            sph_emb = nn.utils.rnn.pack_padded_sequence(
                sph_emb,
                sorted_seq_lengths,
                batch_first=True
            )
            self.audio_rnn_net.flatten_parameters()
            sph_emb, (ht, ct) = self.audio_rnn_net(sph_emb)
            _, desort_index = torch.sort(index, descending=False)
            sph_emb, _ = nn.utils.rnn.pad_packed_sequence(sph_emb, batch_first=True)
            sph_emb = sph_emb[desort_index] 
        # compute asr result
        asr_loss, asr_hyp = self.asr_crit(
            sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        )

        # compute kws location
        kw_pos_mask = self.predict_kw_mask(asr_hyp.transpose(0,1))
        cross_context, cross_att = self.location_cross_att(
            phn_emb, sph_emb, sph_emb, mask=cross_mask.transpose(-2,-1), aux_score=kw_pos_mask
        )
        #cross_context, cross_att = self.location_cross_att(
        #    phn_emb, sph_emb, sph_emb, mask=cross_mask.transpose(-2,-1)
        #)
        kw_det_result = self.det_net(cross_context[:,0,:])
        ## compute alignment loss
        align_loss = self.make_align_loss(
            cross_att, kw_len, sph_len, cross_mask, target
        )

        # compute ctc asr loss
        asr_loss = self.l1 * asr_loss # average batch
        # compute bce detect loss
        det_loss = self.det_crit(kw_det_result, target.to(torch.float32))
        det_loss = self.l2 * det_loss 
        ## compute align loss
        align_loss = self.l3 * torch.mean(align_loss)
        total_loss = asr_loss + det_loss 
        detail_loss = {
            'asr_loss': asr_loss.clone().detach(),
            'det_loss': det_loss.clone().detach(),
            'align_loss': align_loss.clone().detach()
        }
        return total_loss, detail_loss

    @torch.no_grad()
    def evaluate(self, input_data):
        #sph_input, phn_label, kw_label, sph_len, phn_len, kw_len, target = input_data 
        #speech,speech_len,label,label_len,keyword,keyword_len,target
        sph_input, sph_len, phn_label, phn_len, kw_label, kw_len = input_data
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)
        #print (sph_input.size(), phn_label.size(), kw_label.size(), sph_len.size(), phn_len.size(), kw_len.size(), target.size())
        # embedding
        sph_emb = self.au_trans(sph_input)
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
        
        sph_emb, att_score = self.forward_transformer(
            self.au_transformer,
            sph_emb,
            mask=sph_mask,
            cross_embedding=(
                phn_emb, phn_emb, cross_mask
            ),
            analyse=True
        )
        
       
        # compute asr result
        asr_hyp = self.asr_crit.get_hyp(sph_emb)
        asr_loss, asr_hyp = self.asr_crit(
            sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        )
        # compute kws location
        kw_pos_mask = self.predict_kw_mask(asr_hyp.transpose(0,1))
        cross_context, cross_att = self.location_cross_att(
            phn_emb, sph_emb, sph_emb, mask=cross_mask.transpose(-2,-1), aux_score=kw_pos_mask
        )

        kw_det_result = self.det_net(cross_context[:,0,:])
        ## compute alignment loss
        #return sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, target
        #return kw_det_result, kw_label, kw_len, asr_hyp.transpose(0,1), phn_label, sph_len
        return kw_det_result
