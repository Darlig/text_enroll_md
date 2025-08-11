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
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pypinyin import pinyin, Style
import re
import numpy as np

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
        det_result = model.evaluate(input_data)
        det_result = det_result.cpu().numpy()[0]
        return det_result

def test_md(model, wav_scp_path, phone_path, human_label_path, result_label_score_path, sop_token):
    if os.path.exists(result_label_score_path):
        print("Result file already exists: {}. Skip md test.".format(result_label_score_path))
        return
    data_list = load_dataset(wav_scp_path, phone_path, human_label_path)
    print("Loaded {} utterances for testing.".format(len(data_list)))
    with open(result_label_score_path, 'w') as f:
        for n, (uttid, wav_path, phones_int, phones_accuracy) in enumerate(data_list):
            if n % 2000 == 0:
                print("Inferencing {}th utterance: {}".format(n, uttid))
            try:
                det_result = inference(model, wav_path, phones_int, sop_token)
            except Exception as e:
                print(f"Error processing {uttid}: {e}")
                continue
            # print("utt: {}, phones: {}, det_result: {}".format(uttid, phones_int, det_result))
            for i in range(len(det_result)):
                f.write("{}.{}\t{}\t{}\t{}\n".format(uttid, i, phones_accuracy[i], det_result[i], phones_int[i]))


def load_dataset(wav_scp_path, phone_path, human_label_path):
    wav_dict = load_dict_from_file(wav_scp_path)
    phone_dict = load_dict_from_file(phone_path)
    human_score_dict = load_human_score_json(human_label_path)
    data_list = []
    for i, uttid in enumerate(wav_dict.keys()):
        if i % 2000 == 0:
            print(f"Processing {i}th utterance: {uttid}")
        if i > 26350:
            print(f"Processing {i}th utterance: {uttid}")
        assert uttid in human_score_dict, f"Missing human score for {uttid}"
        wav_path = wav_dict[uttid]
        phones = human_score_dict[uttid]['phones']
        phones_accuracy = human_score_dict[uttid]['phones_accuracy']
        is_valid = True
        for ph in phones:
            if ph not in phone_dict:
                # print(f"Phone {ph} not found in phone dictionary")
                is_valid = False
                break
        if not is_valid:
            print(f"Skipping {uttid} due to invalid phones")
            continue

        phones_int = [ phone_dict[ph] for ph in phones if ph != "none" ]

        data_list.append([
            uttid, wav_path, phones_int, phones_accuracy
        ])
    return data_list

def plot_roc_curve(ref_score, hyp_score, test_result_dir, analysis_id, word_py):
    fpr, tpr, thresholds = roc_curve(ref_score, hyp_score)
    # print("fpr tpr thresholds")
    # for i in range(10):
    #     print("{} {} {}".format(fpr[i], tpr[i], thresholds[i]))
    # for i in range(len(fpr) - 20, len(fpr)):
    #     print("{} {} {}".format(fpr[i], tpr[i], thresholds[i]))

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, marker='o', label='ROC curve (area = %0.5f)' % roc_auc)
    #plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for {}'.format(word_py))
    plt.legend(loc="lower right")
    plt_path = os.path.join(test_result_dir, 'unet.transformer_{}_roc.png'.format("{}-{}".format(analysis_id, word_py)))
    plt.savefig(plt_path, dpi=400)
    return roc_auc

def plot_pr_curve_and_analysis(ref_score, hyp_score, test_result_dir, analysis_id, word_py, target_precison):
    # from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(ref_score, hyp_score)
    # print("precision recall thresholds")
    # for i in range(20):
    #     print("{} {} {}".format(precision[i], recall[i], thresholds[i]))
    # for i in range(len(precision) - 10, len(precision) -1):
    #     print("{} {} {}".format(precision[i], recall[i], thresholds[i]))

    # find the threshold that gives the target precision
    target_precision_index = np.argmin(np.abs(precision - target_precison))
    target_recall = recall[target_precision_index]
    target_threshold = thresholds[target_precision_index]
    print("target precision: {}, recall: {}, threshold: {}".format(target_precison, target_recall, target_threshold))

    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for {}'.format(word_py))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt_path = os.path.join(test_result_dir, 'unet.transformer_{}_pr.png'.format("{}-{}".format(analysis_id, word_py)))
    plt.savefig(plt_path, dpi=400)

def plot_sample_distribution(ref_score, hyp_score, test_result_dir, analysis_id, word_py):
    assert len(ref_score) == len(hyp_score)
    hyp_score_pos = [score for i, score in enumerate(hyp_score) if ref_score[i] == 1]
    hyp_score_neg = [score for i, score in enumerate(hyp_score) if ref_score[i] == 0]
    plt.figure()
    plt.hist(hyp_score_pos, bins=50, alpha=0.5, label='Positive Samples', color='blue')
    plt.hist(hyp_score_neg, bins=50, alpha=0.5, label='Negative Samples', color='red')
    plt.xlabel('Hypothesis Score')
    plt.ylabel('Frequency')
    plt.title('Sample Distribution for {}'.format(word_py))
    plt.legend()
    plt_path = os.path.join(test_result_dir, 'unet.transformer_{}_sample_distribution.png'.format("{}-{}".format(analysis_id, word_py)))
    plt.savefig(plt_path, dpi=400)

# def compute_dcf(y_true, y_scores, cost_miss, cost_fa, prior_target, test_result_dir, analysis_id, word_id, roc_auc, f_out_csv):
def compute_dcf(y_true, y_scores, cost_miss, cost_fa, prior_target):
    assert cost_miss > 0 and cost_miss <= 1
    assert cost_fa > 0 and cost_fa <= 1
    assert prior_target > 0 and prior_target < 1

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    print("skip first row of roc_curve with threshold inf")
    fpr, tpr, thresholds_roc = fpr[1:], tpr[1:], thresholds_roc[1:]
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)

    roc_thresh_dict = {}
    for i in range(len(fpr)):
        roc_thresh_dict[thresholds_roc[i]] = (fpr[i], tpr[i])

    # dcf(threshold) = cost_miss * prior_target * p_miss(threshold) + cost_fa * (1 - prior_target) * p_fa(threshold)
    dcf = np.min(cost_miss * prior_target * (1 - tpr) + cost_fa * (1 - prior_target) * fpr)
    dcf_index = np.argmin(cost_miss * prior_target * (1 - tpr) + cost_fa * (1 - prior_target) * fpr)
    dcf_threshold = thresholds_roc[dcf_index]
    best_recall = tpr[dcf_index]
    # best_precision = 1 - fpr[dcf_index]
    dcf_index_pr = np.argmin(abs(dcf_threshold - thresholds_pr))
    best_precision = precision[dcf_index_pr]
    print("cost_miss: {0}, cost_fa: {1}".format(cost_miss, cost_fa))
    print("prior_target: {0}".format(prior_target))
    print("DCF: {0:f}".format(dcf))
    print("DCF threshold: {0}".format(dcf_threshold))
    print("DCF p_miss: {0}".format(1 - tpr[dcf_index]))
    print("DCF p_fa: {0}".format(fpr[dcf_index]))
    print("DCF best recall: {0}".format(best_recall))
    print("DCF best precision: {0}".format(best_precision))
    # print("DCF fa_per_hour: {0}".format(fpr[dcf_index]*shift_per_hour))
    # f_out_csv.write("{}\n".format(", ".join([str(word_id), str(roc_auc), str(cost_miss), str(cost_fa), str(prior_target), str(dcf), str(dcf_threshold), str(1-tpr[dcf_index]), str(fpr[dcf_index]), str(fpr[dcf_index]*shift_per_hour)])))


def result_analysis(result_label_score_path):
    all_results = []
    # y_true = []
    # y_scores = []
    with open(result_label_score_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            phone_index, label, score, phone_id = parts
            all_results.append([phone_index, float(label), float(score), phone_id])
        # reverse human label and model score because a mispronounciation is a positive sample
        y_true = [ 1 - int(res[1]) for res in all_results ]
        # print("length of y_true: {}".format(len(y_true)))
        # print(y_true[:10])
        y_scores = [ 1 - res[2] for res in all_results ]
        # print("length of y_scores: {}".format(len(y_scores)))
        # print(y_scores[:10])
        print(len(y_true), len(y_scores))
    plot_roc_curve(y_true, y_scores, os.path.dirname(result_label_score_path), "analysis", "all")
    plot_pr_curve_and_analysis(y_true, y_scores, os.path.dirname(result_label_score_path), "analysis", "all", 0.21)
    plot_sample_distribution(y_true, y_scores, os.path.dirname(result_label_score_path), "analysis", "all")
    compute_dcf(
        y_true, y_scores, cost_miss=1.0, cost_fa=1.0, prior_target=0.5
    )

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
    result_analysis(result_label_score_path)
