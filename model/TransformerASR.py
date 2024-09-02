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


class TransformerASR(nn.Module):
    def __init__(
        self,
        audio_net_config,
        vocab_net_config,
        predict_token=None,
        num_audio_block=8,
        num_vocab_block=6,
        loss_weight=[0.3,0.6,0.1],
    ):
        super(TransformerASR, self).__init__()

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
                feed_forward=NM.FNNBlock(**au_feed_forward_config)
            ) for _ in range(num_audio_block)
        ])

        # vocab net

        #self.vocab_emb = nn.Embedding(num_token, vocab_transformer_config['size'])

        ## final cross att
        
        self.semantic_token = num_token - 2


        #det_net = [
        #    NM.FNNBlock(**det_config), nn.Linear(au_hidden_dim, 1), nn.Sigmoid()
        #]
        #self.det_net = nn.Sequential(*det_net)

        ctc_conf = {
            'num_tokens': num_token,
            'front_output_size': au_hidden_dim 
        }
        self.l1, self.l2, self.l3 = loss_weight
        self.asr_crit = NM.CTC(**ctc_conf)
        self.det_crit = nn.BCELoss(reduction='mean')
        self.xent = nn.CrossEntropyLoss(reduction='mean')

    def forward_transformer(
        self,
        transformer_module,
        input,
        mask=None,
        cross_embedding=None,
        analyse=False
    ):
        if analyse:
            att_scores = []
        for i, tf_layer in enumerate(transformer_module):
            input, att_score = tf_layer(input, mask, cross_input=cross_embedding)
            if analyse:
                att_scores.append((copy.deepcopy(att_score), copy.deepcopy(input)))
        if analyse:
            return input, att_scores
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
        sph_input, sph_len, phn_label, phn_len, kw_label, kw_len, target = input_data
        b,t,d = sph_input.size()
        #mixspeech, mixspeech_len,label,label_len,keyword,keyword_len,target
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        sph_emb = self.au_trans(sph_input)

        # add position embedding
        sph_emb = self.au_pos_emb(sph_emb)

        sph_emb = self.forward_transformer(
            self.au_transformer,
            sph_emb,
            mask=sph_mask,

        )

        asr_loss, asr_hyp = self.asr_crit(
            sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        )

        ## compute align loss
        total_loss = asr_loss 
        detail_loss = {
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
      
        #print (sph_input.size(), phn_label.size(), kw_label.size(), sph_len.size(), phn_len.size(), kw_len.size(), target.size())
        # embedding
        sph_emb = self.au_trans(sph_input)
       
        # add position embedding
        sph_emb = self.au_pos_emb(sph_emb)
    
      
        sph_emb, att_score = self.forward_transformer(
            self.au_transformer,
            sph_emb,
            mask=sph_mask,
            analyse=True
        )
        asr_loss, asr_hyp = self.asr_crit(
            sph_emb, phn_label, sph_len, phn_len, return_hyp=True
        )
        
        return asr_hyp.transpose(0,1), sph_len, phn_label, asr_hyp
