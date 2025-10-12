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

def inference(model, speech_path, phones_int, is_decode=False, is_gop=False, is_sph_embed=False):
    with torch.no_grad():
        if not is_sph_embed:
            # Load wav file
            wav = torchaudio.load(speech_path)[0]
            # Compute fbank features
            fbank = kaldi.fbank(wav, **FBANK_DEFAULT_SETTING)

            fbank = fbank.unsqueeze(0).to('cuda:0')
            fbank_len = torch.tensor([fbank.size(1)])
            # print("fbank shape: {}".format(fbank.shape))
            speech = fbank
            speech_len = fbank_len
        else:
            sph_embed = np.load(speech_path)
            sph_embed = torch.from_numpy(sph_embed)
            sph_embed = sph_embed.unsqueeze(0).to('cuda:0')
            sph_embed_len = torch.tensor([sph_embed.size(1)])
            speech = sph_embed
            speech_len = sph_embed_len
        phones_int = [int(ph) for ph in phones_int]
        # print("phones_int shape: {}".format(len(phones_int)))
        # print("phones_int: {}".format(phones_int))
        phones_int_org = phones_int.copy()
        # print("phones_int after insert sop token: {}".format(phones_int))
        phones_len = torch.tensor([len(phones_int)])
        phones_int = torch.tensor(phones_int, dtype=torch.int64, device='cuda:0').unsqueeze(0)
        phones_int_org = torch.tensor(phones_int_org, dtype=torch.int64, device='cuda:0').unsqueeze(0)
        # phones_accuracy = torch.tensor(phones_accuracy, dtype=torch.float32, device='cuda:0').unsqueeze(0)

        input = (speech, speech_len, phones_int, phones_len)
        input_data = (d.to('cuda:0') for d in input)
        det_result, hyp_result = model.evaluate(input_data)
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
        det_result = det_result[0]
        hyp_result = hyp_result.cpu().numpy()[0]
        det_result = det_result.cpu().numpy()
        return det_result, hyp_result, asr_result, gop_result


def test_md(model, speech_scp_path, phone_path, human_label_path, result_label_score_path, is_decode=False, is_gop=False, is_sph_embed=False):
    if os.path.exists(result_label_score_path):
        print("Result file already exists: {}. Skip md test.".format(result_label_score_path))
        return
    data_list = load_dataset(speech_scp_path, phone_path, human_label_path)
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
    for n, (uttid, speech_path, phones_int, phones_accuracy) in enumerate(data_list):
        if n % 2000 == 0:
            print("Inferencing {}th utterance: {}".format(n, uttid))
        try:
            det_result, hyp_result, asr_result, gop_result = inference(model, speech_path, phones_int, is_decode, is_gop, is_sph_embed)
        except Exception as e:
            print(f"Error processing {uttid}: {e}")
            continue
        print("utt: {}, phones: {}, det_result: {}".format(uttid, phones_int, det_result))
        for i in range(len(det_result)):
            f_score.write("{}.{}\t{}\t{}\t{}\n".format(uttid, i, phones_accuracy[i], det_result[i], phones_int[i]))
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

def plot_roc_curve_for_paper(ref_score, hyp_score, test_result_dir, analysis_id, word_py):
    """
    Plots a visually enhanced ROC curve suitable for academic papers.

    Args:
        ref_score (list or np.array): True binary labels.
        hyp_score (list or np.array): Target scores, can either be probability estimates
                                     of the positive class or non-thresholded decision values.
        test_result_dir (str): Directory to save the plot.
        analysis_id (str): A unique ID for the analysis, used in the filename.
        word_py (str): The word or concept the plot represents, used in the title.
    """
    fpr, tpr, thresholds = roc_curve(ref_score, hyp_score)
    roc_auc = auc(fpr, tpr)

    plt.style.use('seaborn-v0_8-whitegrid')  # Use a clean grid style
    plt.figure(figsize=(8, 6))  # Adjust figure size for better aspect ratio

    # Plot the ROC curve with enhanced aesthetics
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, linestyle='-', label=f'ROC curve (area = {roc_auc:0.4f})')

    # Plot the random chance line
    plt.plot([0, 1], [0, 1], color='#d62728', lw=2, linestyle='--')

    # Set labels and title with larger font sizes
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Receiver Operating Characteristic for {word_py}', fontsize=14, fontweight='bold')
    
    # Adjust axes limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add a legend and grid
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)

    # Save the figure with high resolution
    plt_path = os.path.join(test_result_dir, f'unet.transformer_{analysis_id}-{word_py}_roc.png')
    plt.savefig(plt_path, dpi=600, bbox_inches='tight') # High DPI and tight bbox for better quality

    print(f"AUC of ROC: {roc_auc:.4f}")
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


def plot_pr_curve_for_paper(ref_score, hyp_score, test_result_dir, analysis_id, word_py, target_precision, epsilon=0):
    """
    Plots a visually enhanced Precision-Recall (PR) curve suitable for academic papers.
    It also identifies and highlights specific points based on a target precision.

    Args:
        ref_score (list or np.array): True binary labels.
        hyp_score (list or np.array): Target scores, can be probability estimates
                                     or non-thresholded decision values.
        test_result_dir (str): Directory to save the plot.
        analysis_id (str): A unique ID for the analysis, used in the filename.
        word_py (str): The word or concept the plot represents, used in the title.
        target_precision (list or float): A list of target precision values to highlight on the curve.
    """
    precision, recall, thresholds = precision_recall_curve(ref_score, hyp_score)

    if not isinstance(target_precision, list):
        target_precision = [target_precision]

    selected_points = []

    print("Target Precision Analysis:")
    print("=" * 60)

    # Plotting setup
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))

    # Plot the main PR curve
    plt.plot(recall, precision, color='#1f77b4', lw=2.5, label='PR Curve')
    
    mask = (precision >= epsilon) & (recall >= epsilon) & (recall <= 1 - epsilon)
        
    # 获取过滤后的索引
    filtered_indices = np.where(mask)[0]
    filtered_precision = precision[filtered_indices]
    # Store and annotate selected points
    for i, target_p in enumerate(target_precision):
        filtered_max_index = np.argmax(filtered_precision)
        max_index = filtered_indices[filtered_max_index]
        max_precision = precision[max_index]
        max_index_recall = recall[max_index]
        print("Max precision: {}, recall: {}".format(max_precision, max_index_recall))
        precision_left = precision[:max_index+1]
        recall_left = recall[:max_index+1]
        thresholds_left = thresholds[:max_index+1]
        precision_right = precision[max_index:]
        recall_right = recall[max_index:]
        thresholds_right = thresholds[max_index:]
        
        closest_precision_index_left = np.argmin(np.abs(precision_left - target_p))
        selected_precision_left = precision_left[closest_precision_index_left]
        selected_recall_left = recall_left[closest_precision_index_left]
        selected_threshold_left = thresholds_left[closest_precision_index_left] if closest_precision_index_left < len(thresholds_left) else thresholds_left[-1]
        closest_precision_index_right = np.argmin(np.abs(precision_right - target_p))
        selected_precision_right = precision_right[closest_precision_index_right]
        selected_recall_right = recall_right[closest_precision_index_right]
        selected_threshold_right = thresholds_right[closest_precision_index_right] if closest_precision_index_right < len(thresholds_right) else thresholds_right[-1]
        
        if selected_recall_left == selected_recall_right:
            selected_precisions = [selected_precision_left]
            selected_recalls = [selected_recall_left]
            selected_thresholds = [selected_threshold_left]
        else:
            selected_precisions = [selected_precision_left, selected_precision_right]
            selected_recalls = [selected_recall_left, selected_recall_right]
            selected_thresholds = [selected_threshold_left, selected_threshold_right]

        for i, selected_precision in enumerate(selected_precisions):
            selected_recall = selected_recalls[i]
            selected_threshold = selected_thresholds[i]
            
            target_f1 = 2 * selected_precision * selected_recall / (selected_precision + selected_recall + 1e-12)

            point_data = {
                'target_precision': target_p,
                'actual_precision': selected_precision,
                'recall': selected_recall,
                'f1': target_f1,
                'threshold': selected_threshold
            }
            selected_points.append(point_data)

            print(f"Target P={target_p:.4f} -> Actual P={selected_precision:.4f}, R={selected_recall:.4f}, F1={target_f1:.4f}, Threshold={selected_threshold:.6f}")

            ## Add an annotation for each selected point
            #plt.scatter([selected_recall], [selected_precision], s=120, color='red', marker='X', zorder=5)
            #plt.annotate(
            #    f'P={selected_precision:.2f}, R={selected_recall:.2f}\n(Thr={selected_threshold:.3f})',
            #    (selected_recall, selected_precision),
            #    textcoords="offset points",
            #    xytext=(15, 15),
            #    ha='right',
            #    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
            #    fontsize=10
            #)
    
    # Set plot title and labels with enhanced font styles
    plt.title(f'Precision-Recall Curve for {word_py}', fontsize=14, fontweight='bold')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right", fontsize=10)

    # Save the figure
    plt_path = os.path.join(test_result_dir, f'unet.transformer_{analysis_id}-{word_py}_pr.png')
    plt.savefig(plt_path, dpi=600, bbox_inches='tight')

    return selected_points


from scipy.stats import norm

def plot_det_curve(ref_score, hyp_score, test_result_dir, analysis_id, word_py):
    """
    Plots a Detection Error Tradeoff (DET) curve.

    Args:
        ref_score (list or np.array): True binary labels.
        hyp_score (list or np.array): Target scores, can be probability estimates
                                     or non-thresholded decision values.
        test_result_dir (str): Directory to save the plot.
        analysis_id (str): A unique ID for the analysis, used in the filename.
        word_py (str): The word or concept the plot represents, used in the title.
    """
    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
    # from these, we can derive FAR and FRR
    fpr, tpr, _ = roc_curve(ref_score, hyp_score)

    # Convert to DET curve axes: FAR vs. FRR
    # FAR is the same as FPR
    # FRR = 1 - TPR
    far = fpr
    frr = 1 - tpr

    # Convert the rates to a normal deviate scale
    # We use a custom function to handle 0 and 1 values gracefully
    def to_normal_deviate(p):
        p = np.clip(p, 1e-10, 1 - 1e-10) # Clip values to avoid issues with infinity
        return norm.ppf(p)
    
    far_norm = to_normal_deviate(far)
    frr_norm = to_normal_deviate(frr)

    # Plotting setup
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))

    # Plot the DET curve
    plt.plot(far_norm, frr_norm, color='#2ca02c', lw=2.5, label='DET Curve')

    # Find and plot the Equal Error Rate (EER) point
    # EER is the point where FAR equals FRR
    eer_index = np.argmin(np.abs(far - frr))
    eer_far = far[eer_index]
    eer_frr = frr[eer_index]
    eer_point = (to_normal_deviate(eer_far), to_normal_deviate(eer_frr))
    
    plt.scatter([eer_point[0]], [eer_point[1]], s=120, color='red', marker='X', zorder=5)
    plt.annotate(
        f'EER = {eer_far:.2%}', # Format as percentage
        eer_point,
        textcoords="offset points",
        xytext=(15, -15),
        ha='right',
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
        fontsize=10
    )
    print(f"Equal Error Rate (EER): {eer_far:.4f}")

    # Set custom tick labels for the normal deviate scale
    tick_values = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    tick_labels = [f'{p:.1%}' for p in tick_values]
    plt.xticks(to_normal_deviate(tick_values), tick_labels)
    plt.yticks(to_normal_deviate(tick_values), tick_labels)

    # Set plot title and labels with enhanced font styles
    plt.title(f'Detection Error Tradeoff (DET) for {word_py}', fontsize=14, fontweight='bold')
    plt.xlabel('False Accept Rate (FAR)', fontsize=12)
    plt.ylabel('False Reject Rate (FRR)', fontsize=12)
    plt.grid(True, which='both', linestyle='--')
    plt.legend(loc="upper right", fontsize=10)

    # Save the figure
    plt_path = os.path.join(test_result_dir, f'unet.transformer_{analysis_id}-{word_py}_det.png')
    plt.savefig(plt_path, dpi=600, bbox_inches='tight')

    return eer_far


from scipy.stats import norm



def plot_det_curve_for_paper(ref_score, hyp_score, test_result_dir, analysis_id, word_py):
    """
    Plots a visually enhanced DET (Detection Error Tradeoff) curve suitable for academic papers.
    
    Args:
        ref_score (list or np.array): True binary labels.
        hyp_score (list or np.array): Target scores, can either be probability estimates
                                     of the positive class or non-thresholded decision values.
        test_result_dir (str): Directory to save the plot.
        analysis_id (str): A unique ID for the analysis, used in the filename.
        word_py (str): The word or concept the plot represents, used in the title.
    
    Returns:
        tuple: (min_dcf, eer) - Minimum Detection Cost Function and Equal Error Rate
    """
    # Calculate FPR and TPR using ROC curve
    fpr, tpr, thresholds = roc_curve(ref_score, hyp_score)
    
    # Calculate False Negative Rate (Miss Rate)
    fnr = 1 - tpr
    
    # Convert to percentages
    fpr_percent = fpr * 100
    fnr_percent = fnr * 100
    
    # Apply normal inverse transformation for DET curve
    # Add small epsilon to avoid infinite values at 0 and 1
    epsilon = 1e-6
    fpr_norm = np.maximum(epsilon, np.minimum(1-epsilon, fpr))
    fnr_norm = np.maximum(epsilon, np.minimum(1-epsilon, fnr))
    
    # Normal inverse transformation
    fpr_norm_inv = norm.ppf(fpr_norm)
    fnr_norm_inv = norm.ppf(fnr_norm)
    
    # Calculate Equal Error Rate (EER)
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
    
    # Calculate minimum Detection Cost Function (assuming equal costs and priors)
    # DCF = P_miss * P_target + P_fa * (1 - P_target)
    # For equal priors (P_target = 0.5) and equal costs: DCF = (P_miss + P_fa) / 2
    # dcf = (fnr + fpr) / 2
    # min_dcf = np.min(dcf) * 100
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    
    # # Plot the DET curve with enhanced aesthetics
    plt.plot(fpr_norm_inv, fnr_norm_inv, color='#1f77b4', lw=2.5, linestyle='-', 
             label=f'DET curve (EER = {eer:.2f}%)')
    
    # Mark the EER point
    plt.plot(fpr_norm_inv[eer_idx], fnr_norm_inv[eer_idx], 'ro', markersize=8, 
             label=f'EER = {eer:.2f}%')
    
    # Set custom tick positions and labels for better readability
    tick_positions = norm.ppf([0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    tick_labels = ['10', '20', '40', '50', '60', '70', '80', '90']
    
    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)
    
    # Set labels and title with larger font sizes
    plt.xlabel('False Positive Rate (%)', fontsize=12)
    plt.ylabel('False Negative Rate (Miss Rate) (%)', fontsize=12)
    plt.title(f'Detection Error Tradeoff (DET) Curve for {word_py}', fontsize=14, fontweight='bold')
    
    # Set axis limits
    plt.xlim([norm.ppf(0.001), norm.ppf(0.99)])
    plt.ylim([norm.ppf(0.001), norm.ppf(0.99)])
    
    # Add a legend and grid
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the figure with high resolution
    plt_path = os.path.join(test_result_dir, f'unet.transformer_{analysis_id}-{word_py}_det.png')
    plt.savefig(plt_path, dpi=600, bbox_inches='tight')
    
    print(f"Equal Error Rate (EER): {eer:.2f}%")
    # print(f"Minimum DCF: {min_dcf:.2f}%")
    
    plt.close()  # Close the figure to free memory
    
    return eer

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


def result_analysis(result_label_score_path, is_gop=False, target_precision=0, pr_epsilon=0):
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
    plot_roc_curve_for_paper(y_true, y_scores, os.path.dirname(result_label_score_path), analysis_id, "all")
    selected_points = plot_pr_curve_for_paper(y_true, y_scores, os.path.dirname(result_label_score_path), analysis_id, "all", target_precision, epsilon=pr_epsilon)
    plot_det_curve_for_paper(y_true, y_scores, os.path.dirname(result_label_score_path), analysis_id, "all")
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
    keyword_config = train_config['data_config']['keyword_config']['config']
    result_label_score_path = test_config['result_label_score']
    target_precision = test_config.get('target_precision', 0.21)
    default_target_level = ['phone']
    default_special_token = {'sop': 1}
    is_sph_embed = True if 'sph_emb_scp' in test_config else False
    speech_scp = test_config['wav_scp'] if not is_sph_embed else test_config['sph_emb_scp']
    pr_epsilon = test_config.get('pr_epsilon', 0.03)

    if not os.path.exists(os.path.dirname(result_label_score_path)):
        os.makedirs(os.path.dirname(result_label_score_path))

    test_model = load_model()
    test_model.eval()
    test_md(
        test_model,
        speech_scp,
        test_config['phone'],
        test_config['human_label'],
        result_label_score_path,
        is_decode=test_config.get('is_decode', False),
        is_gop=test_config.get('is_gop', False),
        is_sph_embed=is_sph_embed
    )
    result_analysis(result_label_score_path, target_precision=target_precision, pr_epsilon=pr_epsilon)
    if test_config.get('is_gop', False):
        gop_result_label_score_path = os.path.join(os.path.dirname(result_label_score_path), 'result_gop.txt')
        result_analysis(gop_result_label_score_path, is_gop=True, target_precision=target_precision, pr_epsilon=pr_epsilon)
