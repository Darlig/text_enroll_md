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

def insert_special_token(phone_seq, target_level, special_token):
    # print("length of phone_seq: {}".format(len(phone_seq)))
    kw_spec_mask = [0] * len(phone_seq)
    if 'phone' in target_level:
        sop = special_token['sop']
        for i in range(len(phone_seq)):
            phone_seq.insert(i*2, sop)
            kw_spec_mask.insert(i*2, 1)
    if 'word' in target_level:
        psok = special_token['psok']
        phone_seq.insert(0, psok)
        kw_spec_mask.insert(0, 1)
    return phone_seq, kw_spec_mask

def result_extract(det_result, target_level):
    if 'word' in target_level:
        det_result = det_result[1:]
    return det_result

def inference(model, wav_path, phones_int, is_decode=False, is_gop=False):
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
        phones_int_org = phones_int.copy()
        # phones_int, kw_spec_mask = insert_special_token(phones_int, target_level, special_token)
        # print("phones_int after insert sop token: {}".format(phones_int))
        # phones_len = torch.tensor([len(phones_int)])
        # phones_int = torch.tensor(phones_int, dtype=torch.int64, device='cuda:0').unsqueeze(0)
        phones_int_org = torch.tensor(phones_int_org, dtype=torch.int64, device='cuda:0').unsqueeze(0)
        # kw_spec_mask = torch.tensor(kw_spec_mask, dtype=torch.int64, device='cuda:0').unsqueeze(0)
        # phones_accuracy = torch.tensor(phones_accuracy, dtype=torch.float32, device='cuda:0').unsqueeze(0)

        input = (fbank, fbank_len)
        input_data = (d.to('cuda:0') for d in input)
        hyp_result = model.evaluate(input_data)
        if is_decode:
            asr_result = model.greedy_decode(hyp_result)
            asr_result = asr_result[0]
        else:
            asr_result = None
        if is_gop:
            gop_result = model.compute_gop(hyp_result, phones_int_org)
            # print("gop_result: ", gop_result)
        else:
            gop_result = None
        # det_result = det_result[0]
        hyp_result = hyp_result.cpu().numpy()[0]
        # det_result = result_extract(det_result, target_level)
        # det_result = det_result.cpu().numpy()
        return hyp_result, asr_result, gop_result


def test_md(model, wav_scp_path, phone_path, human_label_path, result_label_score_path, is_decode=False, is_gop=False):
    if os.path.exists(result_label_score_path):
        print("Result file already exists: {}. Skip md test.".format(result_label_score_path))
        return
    data_list = load_dataset(wav_scp_path, phone_path, human_label_path)
    print("Loaded {} utterances for testing.".format(len(data_list)))
    result_dir = os.path.dirname(result_label_score_path)
    result_decode_path = os.path.join(result_dir, 'result_decode.txt')
    reference_path = os.path.join(result_dir, 'reference.txt')
    result_gop_path = os.path.join(result_dir, 'result_gop.txt')
    f_score = open(result_label_score_path, 'w')
    if is_decode:
        f_decode = open(result_decode_path, 'w')
        f_refer = open(reference_path, 'w')
    if is_gop:
        f_gop = open(result_gop_path, 'w')
    for n, (uttid, wav_path, phones_int, phones_accuracy) in enumerate(data_list):
        if n % 2000 == 0:
            print("Inferencing {}th utterance: {}".format(n, uttid))
        try:
            hyp_result, asr_result, gop_result = inference(model, wav_path, phones_int, is_decode, is_gop)
        except Exception as e:
            print(f"Error processing {uttid}: {e}")
            continue
        # print("utt: {}, phones: {}, det_result: {}".format(uttid, phones_int, det_result))
        # for i in range(len(det_result)):
        #     f_score.write("{}.{}\t{}\t{}\t{}\n".format(uttid, i, phones_accuracy[i], det_result[i], phones_int[i]))
        if is_decode:
            asr_result_str = " ".join([ str(p) for p in asr_result ])
            human_phn_str = " ".join(phones_int)
            f_decode.write("{} {}\n".format(uttid, asr_result_str))
            f_refer.write("{} {}\n".format(uttid, human_phn_str))
        if is_gop:
            for i in range(len(gop_result)):
                f_gop.write("{}.{}\t{}\t{}\t{}\n".format(uttid, i, phones_accuracy[i], gop_result[i]["gop"], phones_int[i]))
    f_score.close()
    if is_decode:
        f_decode.close()
        f_refer.close()


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
    print("AUC of ROC: {}".format(roc_auc))
    return roc_auc

def plot_pr_curve_and_analysis(ref_score, hyp_score, test_result_dir, analysis_id, word_py, target_precision):
    # from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(ref_score, hyp_score)
    # print("precision recall thresholds")
    # for i in range(20):
    #     print("{} {} {}".format(precision[i], recall[i], thresholds[i]))
    # for i in range(len(precision) - 10, len(precision) -1):
    #     print("{} {} {}".format(precision[i], recall[i], thresholds[i]))

    # 确保target_precision是list格式
    if not isinstance(target_precision, list):
        target_precision = [target_precision]
    
    # 存储所有选中的点
    selected_points = []
    
    print("Target Precision Analysis:")
    print("=" * 60)
    
    # 遍历每个目标precision
    for i, target_p in enumerate(target_precision):
        # find the threshold that gives the target precision
        closest_precision_index = np.argmin(np.abs(precision - target_p))
        selected_precision = precision[closest_precision_index]
        selected_recall = recall[closest_precision_index]
        selected_threshold = thresholds[closest_precision_index]
        target_f1 = 2 * selected_precision * selected_recall / (selected_precision + selected_recall + 1e-12)
        
        # 存储点信息
        selected_points.append({
            'target_precision': target_p,
            'actual_precision': selected_precision,
            'recall': selected_recall,
            'f1': target_f1,
            'threshold': selected_threshold
        })
        
        print("Target P={:.4f} -> Actual P={:.4f}, R={:.4f}, F1={:.4f}, Threshold={:.6f}".format(
            target_p, selected_precision, selected_recall, target_f1, selected_threshold))

    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, marker='o', label='PR Curve')
    
    # 为每个选中的点画散点
    colors = ['red', 'blue', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    for i, point in enumerate(selected_points):
        color = colors[i % len(colors)]
        marker = ['x', 'o', 's', '^', 'v', 'D', '*', '+', '<', '>'][i % 10]
        plt.scatter([point['recall']], [point['actual_precision']], 
                   color=color, marker=marker, s=100, 
                   label='Target P={:.2f} (P={:.2f}, R={:.2f})'.format(
                       point['target_precision'], point['actual_precision'], point['recall']))
    
    plt.legend(loc="lower left", fontsize=8)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for {}'.format(word_py))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt_path = os.path.join(test_result_dir, 'unet.transformer_{}_pr.png'.format("{}-{}".format(analysis_id, word_py)))
    plt.savefig(plt_path, dpi=400)
    
    return selected_points

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
def compute_dcf(y_true, y_scores, cost_miss, cost_fa, prior_target, analysis_id):
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
    print("{} DCF:".format(analysis_id))
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


def result_analysis(result_label_score_path, is_gop=False, target_precision=0.21):
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
    analysis_id = "analysis" if not is_gop else "gop_analysis"
    plot_roc_curve(y_true, y_scores, os.path.dirname(result_label_score_path), analysis_id, "all")
    selected_points = plot_pr_curve_and_analysis(y_true, y_scores, os.path.dirname(result_label_score_path), analysis_id, "all", target_precision)
    plot_sample_distribution(y_true, y_scores, os.path.dirname(result_label_score_path), analysis_id, "all")
    compute_dcf(
        y_true, y_scores, cost_miss=1.0, cost_fa=1.0, prior_target=0.5, analysis_id=analysis_id
    )
    return selected_points

if __name__ == '__main__':

    test_config_path = sys.argv[1]

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    test_config = yaml.load(open(test_config_path), Loader=yaml.FullLoader)
    trained_ckpt_path = test_config['trained_ckpt_path']
    train_config = test_config.get('train_config', None)
    # sop_token = train_config['data_config']['keyword_config']['config']['special_token']['sop']
    # keyword_config = train_config['data_config']['keyword_config']['config']
    result_label_score_path = test_config['result_label_score']
    target_precision = test_config.get('target_precision', 0.21)
    # default_target_level = ['phone']
    # default_special_token = {'sop': 1}
    # if 'target_level' in keyword_config:
    #     target_level = keyword_config['target_level']
    # else:
    #     print("No target_level in keyword_config, use default: {}".format(default_target_level))
    #     target_level = default_target_level
    # if 'special_token' in keyword_config:
    #     special_token = keyword_config['special_token']
    # else:
    #     print("No special_token in keyword_config, use default: {}".format(default_special_token))
    #     special_token = default_special_token
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
        is_decode=test_config.get('is_decode', False),
        is_gop=test_config.get('is_gop', False)
    )
    # result_analysis(result_label_score_path)
    if test_config.get('is_gop', False):
        gop_result_label_score_path = os.path.join(os.path.dirname(result_label_score_path), 'result_gop.txt')
        result_analysis(gop_result_label_score_path, is_gop=True, target_precision=target_precision)
