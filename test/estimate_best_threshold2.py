import sys
import numpy as np
from sklearn.metrics import roc_curve
import argparse

from result_analysis import ignore_keyword, load_pos_keyword, load_neg_keyword, load_pos_refer, load_pos_scores, load_neg_scores

in_pos_result = sys.argv[1]
in_pos_refer = sys.argv[2]
in_pos_keyword = sys.argv[3]
in_pos_id = sys.argv[4]
in_neg_result_scp = sys.argv[5]
in_neg_keyword = sys.argv[6]
cost_miss = float(sys.argv[7])
cost_fa = float(sys.argv[8])
prior_target = float(sys.argv[9])
out_csv = sys.argv[10]

shift_per_hour = 10 * 3600

def compute_dcf(y_true, y_scores, cost_miss, cost_fa, prior_target, out_csv):
    assert cost_miss > 0 and cost_miss <= 1
    assert cost_fa > 0 and cost_fa <= 1
    assert prior_target > 0 and prior_target < 1

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    print("skip first row of roc_curve with threshold inf")
    fpr, tpr, thresholds_roc = fpr[1:], tpr[1:], thresholds_roc[1:]
    # print("first row of roc_curve: {} {} {}".format(fpr[0], tpr[0], thresholds_roc[0]))
    roc_thresh_dict = {}
    for i in range(len(fpr)):
        roc_thresh_dict[thresholds_roc[i]] = (fpr[i], tpr[i])

    # dcf(threshold) = cost_miss * prior_target * p_miss(threshold) + cost_fa * (1 - prior_target) * p_fa(threshold)
    dcf = np.min(cost_miss * prior_target * (1 - tpr) + cost_fa * (1 - prior_target) * fpr)
    dcf_index = np.argmin(cost_miss * prior_target * (1 - tpr) + cost_fa * (1 - prior_target) * fpr)
    dcf_threshold = thresholds_roc[dcf_index]
    print("cost_miss: {0}, cost_fa: {1}".format(cost_miss, cost_fa))
    print("prior_target: {0}".format(prior_target))
    print("DCF: {0:f}".format(dcf))
    print("DCF threshold: {0}".format(dcf_threshold))
    print("DCF p_miss: {0}".format(1 - tpr[dcf_index]))
    print("DCF p_fa: {0}".format(fpr[dcf_index]))
    print("DCF fa_per_hour: {0}".format(fpr[dcf_index]*shift_per_hour))
    #with open(out_csv, 'w') as f_out_csv:
        #f_out_csv.write("{}".format(dcf_threshold))
    f_out_csv.write("\n{}".format(", ".join([in_pos_id, in_neg_keyword, str(cost_miss), str(cost_fa), str(prior_target), str(dcf), str(dcf_threshold), str(1-tpr[dcf_index]), str(fpr[dcf_index]), str(fpr[dcf_index]*shift_per_hour)])))
    #f_out_csv.write("\n{}".format(", ".join([in_pos_id, in_neg_keyword, str(cost_miss), str(cost_fa), str(prior_target), str(dcf), str(dcf_threshold), str(1-tpr[dcf_index]), str(fpr[dcf_index])])))

pos_keyword_dict = load_pos_keyword(in_pos_keyword, ignore_keyword)
neg_ids = load_neg_keyword(in_neg_keyword, ignore_keyword)
pos_refer_dict = load_pos_refer(in_pos_refer, pos_keyword_dict)
pos_scores = load_pos_scores(in_pos_result, pos_refer_dict)

neg_result_dict = {}
with open(in_neg_result_scp) as f_neg_result_scp:
    for line in f_neg_result_scp.readlines():
        neg_result_id, neg_result_path = line.strip().split()
        neg_result_dict[neg_result_id] = neg_result_path

f_out_csv = open(out_csv, 'w')
f_out_csv.write("{}".format(", ".join(["pos_id", "neg_keyword", "cost_miss", "cost_fa", "prior", "dcf", "dcf_threshold", "fnr", "fpr", "fa_per_hour"])))
#f_out_csv.write("{}".format(", ".join(["pos_id", "neg_keyword", "cost_miss", "cost_fa", "prior", "dcf", "dcf_threshold", "fnr", "fpr"])))
for neg_result_id in neg_result_dict:
    neg_result = neg_result_dict[neg_result_id]
    neg_scores = load_neg_scores(neg_result, neg_ids)
    #neg_scores = load_neg_scores(neg_result)

    y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    y_scores = np.array(pos_scores + neg_scores)


    #fpr, fnr, thresholds = det_curve(y_true, y_scores)
    compute_dcf(y_true, y_scores, cost_miss, cost_fa, prior_target, out_csv)
f_out_csv.close()

def load_scores(gop_score_result, human_result):
    dtype = str
    gop_data = np.loadtxt(gop_score_result, delimiter='\t', dtype=dtype)
    y_scores = gop_data[:, 1].astype(float)

    human_data = np.loadtxt(human_result, delimiter='\t', dtype=dtype)
    human_dict = {}
    for i in range(len(human_data)):
        human_dict[human_data[i, 0]] = human_data[i, 1]
    y_true = []
    for i in range(len(gop_data)):
        key = gop_data[i, 0]
        assert key in human_dict
        y_true.append(human_dict[key])
    y_true = np.array(y_true, dtype=float)
  
    assert len(y_true) == len(y_scores)

    #Scores check
    out_of_range = np.any((y_scores < 0) | (y_scores> 1))
    # If out_of_range is True, it means there are values not in the range [0, 1]
    if out_of_range:
        print("err There are values in y_scores_exp that are not in the range [0, 1].")
        sys.exit()

    # Exchange sample labels and corresponding y_scores value
    # In MD  incorrect pronunciation is positive , correct pronunciation is negative
    # y_true = 1 - y_true
    # y_scores = 1 - y_scores

    return y_true, y_scores

