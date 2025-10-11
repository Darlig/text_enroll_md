# text enroll MD

mispronunciation detection 模型训练项目

## 环境准备
```bash
conda env create -f environment.yaml
conda activate text_enroll_md
```

## 入口代码
### 202510 TIMIT训练(@dragon03)
```bash
conda activate /work104/weiyang/environment/anaconda3/envs/text_enroll_md
bash run_train_keyword_init_final.sh train_kw_init_fin.py --config /work104/weiyang/project/maolidan_thesis/experiment/text_enroll_md/config/train_config/TransformerKWS_nocross/timit_embed_pos0.1_change0.2_transformer_4kw_4concat_ctc0.1_det0.9.yaml --GPU 0,1,2,3,4 --port 22130
```

### 202510 aishell-2训练(@dragon03)
```bash
conda activate /work104/weiyang/environment/anaconda3/envs/text_enroll_md
bash run_train_keyword_init_final.sh train_kw_init_fin.py --config /work104/weiyang/project/maolidan_thesis/experiment/text_enroll_md/config/train_config/TransformerKWS_nocross/aishell2_cut_gt400_kaldi_lex2_tone_whole_utt_neg_by_lex_pos0.01_change0.8_w0.1_0.1_0.1_0.7_spec_mask_w_psok_hanlp_transformer_4sph_4kw_4concat_ctc_det_reverb_perturb_nomix_keyword_init_final_seg_pos_spec0.8.yaml --GPU 0,1,2,3,4 --port 22130
```

