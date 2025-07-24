import os
import yaml
import json
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import sys

from yamlinclude import YamlIncludeConstructor
# from local.utils import make_dict_from_file
from model import m_dict
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pypinyin import pinyin, Style
import re

FBANK_DEFAULT_SETTING = {
    'num_mel_bins': 40, 'frame_length': 25, 'frame_shift': 10
}


def load_model():
    trained_ckpt = torch.load(trained_ckpt_path)
    trained_ckpt = trained_ckpt['model']

    model_arch = train_config['model_arch']
    model_config = train_config['model_config']
    model = m_dict[model_arch]

    test_model = model(**model_config)
    test_model.load_state_dict(trained_ckpt)
    test_model.to('cuda:0')
    test_model.eval()
    return test_model

def load_dict_from_file(file_path):
    new_dict = {}
    with open(file_path) as f:
        for line in f:
            # print("line: ", line.strip())
            key, value = re.split(r'\s+', line.strip(), maxsplit=1)
            new_dict[key] = value
    return new_dict


    #  1 {
    #  2     "c0860096": {
    #  3         "text": "钟",
    #  4         "words": [
    #  5             {
    #  6                 "text": "钟",
    #  7                 "phones": [
    #  8                     "zh",
    #  9                     "ong1"
    # 10                 ],
    # 11                 "phones-accuracy": [
    # 12                     1,
    # 13                     1
    # 14                 ]
    # 15             }
    # 16         ]
    # 17     },

def load_human_score_json(file_path):
    human_labels_dict = {}
    with open(file_path) as f:
        human_score_dict = json.load(f)
        for uttid in human_score_dict:
            words = human_score_dict[uttid]['words']
            human_labels_dict[uttid] = {
                'phones': [ ph for word in words for ph in word['phones'] ],
                'phones_accuracy': [ ph_acc for word in words for ph_acc in word['phones-accuracy'] ],   
            }
    return human_labels_dict

def insert_special_token(phone_seq, sop_token):
    # print("length of phone_seq: {}".format(len(phone_seq)))
    for i in range(len(phone_seq)):
        phone_seq.insert(i*2, sop_token)
    return phone_seq

def inference(model, wav_path, phones_int, sop_token):
    with torch.no_grad():
        # Load wav file
        wav = torchaudio.load(wav_path)[0]
        # Compute fbank features
        fbank = kaldi.fbank(wav, **FBANK_DEFAULT_SETTING)

        fbank = fbank.unsqueeze(0).to('cuda:0')
        fbank_len = torch.tensor([fbank.size(1)])
        # print("fbank shape: {}".format(fbank.shape))
        phones_int = [int(ph) for ph in phones_int]
        # print("phones_int shape: {}".format(len(phones_int)))
        # print("phones_int: {}".format(phones_int))
        phones_int = insert_special_token(phones_int, sop_token)
        # print("phones_int after insert sop token: {}".format(phones_int))
        phones_len = torch.tensor([len(phones_int)])
        phones_int = torch.tensor(phones_int, dtype=torch.int64, device='cuda:0').unsqueeze(0)
        # phones_accuracy = torch.tensor(phones_accuracy, dtype=torch.float32, device='cuda:0').unsqueeze(0)

        input = (fbank, fbank_len, phones_int, phones_len)
        input_data = (d.to('cuda:0') for d in input)
        det_result, hyp_result = model.evaluate(input_data)
        det_result = det_result.cpu().numpy()[0]
        hyp_result = hyp_result.cpu().numpy()[0]
        return det_result, hyp_result

def test_md(model, wav_scp_path, phone_path, human_label_path, result_label_score_path, sop_token):
    data_list = load_dataset(wav_scp_path, phone_path, human_label_path)
    with open(result_label_score_path, 'w') as f:
        for uttid, wav_path, phones_int, phones_accuracy in data_list:
            det_result, hyp_result = inference(model, wav_path, phones_int, sop_token)
            # print("utt: {}, phones: {}, det_result: {}".format(uttid, phones_int, det_result))
            for i in range(len(det_result)):
                f.write("{}.{}\t{}\t{}\n".format(uttid, i, phones_accuracy[i], det_result[i], phones_int[i]))
            



def load_dataset(wav_scp_path, phone_path, human_label_path):
    wav_dict = load_dict_from_file(wav_scp_path)
    phone_dict = load_dict_from_file(phone_path)
    human_score_dict = load_human_score_json(human_label_path)
    data_list = []
    for uttid in wav_dict:
        assert uttid in human_score_dict, f"Missing human score for {uttid}"
        wav_path = wav_dict[uttid]
        phones = human_score_dict[uttid]['phones']
        phones_accuracy = human_score_dict[uttid]['phones_accuracy']
        phones_int = [ phone_dict[ph] for ph in phones ]
        data_list.append([
            uttid, wav_path, phones_int, phones_accuracy
        ])
    return data_list

if __name__ == '__main__':

    test_config_path = sys.argv[1]

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    test_config = yaml.load(open(test_config_path), Loader=yaml.FullLoader)
    trained_ckpt_path = test_config['trained_ckpt_path']
    train_config = test_config.get('train_config', None)
    sop_token = train_config['data_config']['keyword_config']['config']['special_token']['sop']
    result_label_score_path = test_config['result_label_score']
    if not os.path.exists(os.path.dirname(result_label_score_path)):
        os.makedirs(os.path.dirname(result_label_score_path))

    test_model = load_model()
    test_model.eval()
    test_md(
        test_model,
        test_config['wav_scp'],
        test_config['phone'],
        test_config['human_label'],
        result_label_score_path,
        sop_token
    )
    # result_analysis()
