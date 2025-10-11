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

