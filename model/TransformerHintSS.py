import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import model.NetModules as NM

att_dict = {
    'MultiHeadCrossAtt': NM.MultiHeadCrossAtt, 
    'MultiHeadAtt': NM.MultiHeadAtt
}


class TransformerHintSS(nn.Module):
    def __init__(
        self,
        audio_net_config,
        vocab_net_config,
        predict_token=None,
        num_audio_block=8,
        num_vocab_block=6,
        loss_weight=[0.3,0.6,0.1],
    ):
        super(TransformerHintSS, self).__init__()

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

        ctc_conf = {
           'num_tokens': num_token,
           'front_output_size': au_hidden_dim
        }
        self.asr_crit = NM.CTC(**ctc_conf)

        res_net = [
            nn.Linear(au_hidden_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 200), nn.Sigmoid()
        ]
        self.res_net = nn.Sequential(*res_net)
        #self.crit = nn.MSELoss(reduction='none')
        self.crit = nn.MSELoss()

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
        #sph_input, phn_label, kw_label, sph_len, phn_len, kw_len, target = input_data 
        sph_input, sph_len, clean, clean_len, phn_label,phn_len, kw_label, kw_len, target = input_data
        #sph_input, sph_len, phn_label, phn_len, kw_label, kw_len = input_data
        b,t,d = sph_input.size()
        #mixspeech, mixspeech_len,label,label_len,keyword,keyword_len,target
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)
        loss_mask = ~NM.make_mask(sph_len)
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
        sph_emb = self.forward_transformer(
            self.au_transformer,
            sph_emb,
            mask=sph_mask,
            cross_embedding=(
                phn_emb, phn_emb, cross_mask
            )
        )

        asr_loss, asr_hyp = self.asr_crit(
             sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        )

        b,t,d = sph_emb.size()
        loss_mask = loss_mask.eq(0)
        sph_rec_mask = self.res_net(sph_emb.view(b*t, d))
        sph_rec = sph_rec_mask.view(b,t,-1) * sph_input
        rec_loss = self.crit(sph_rec, clean)
        rec_loss = rec_loss.mean()
        #rec_loss = rec_loss.masked_fill(loss_mask, 0)
        #rec_loss = rec_loss.mean()
        ## compute align loss
        total_loss = rec_loss + asr_loss
        #total_loss = rec_loss

        detail_loss = {
            'mse_loss': rec_loss.clone().detach(),
            'asr_loss': asr_loss.clone().detach(),
        }
        return total_loss, detail_loss
    
    def generate_mask(self, sequence_length, batch_size, p=0.5):
        mask = torch.tensor(torch.bernoulli(torch.ones(batch_size, sequence_length) * p), dtype=torch.float32)
        return mask

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
