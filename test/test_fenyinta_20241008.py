import os
import yaml
import json
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import sys

from yamlinclude import YamlIncludeConstructor
from local.utils import make_dict_from_file
from model import m_dict
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pypinyin import pinyin, Style

test_config_file = sys.argv[1]

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
test_config = yaml.load(open(test_config_file), Loader=yaml.FullLoader)
testset_dir = test_config['testset_dir']
train_config = test_config.get('train_config', None)
train_datalist = train_config['data_config']['data_list']
train_data_dir = os.path.dirname(train_datalist)
test_result_dir = test_config['test_result_dir']
exp_dir = train_config['exp_config']['exp_dir']
trained_ckpt_path = os.path.join(exp_dir, 'kwatt_asr_avg.pt')
if 'trained_ckpt' in test_config:
    trained_ckpt_path = test_config['trained_ckpt']

word2id = make_dict_from_file(train_data_dir+'/word2id.txt')
id2word = {int(v): k for k, v in word2id.items()}
chunk_shift = test_config['chunk_shift']
chunk_size = test_config['chunk_size']
plot_each_word = test_config.get('plot_each_word', False)
test_option = test_config['test_option']
pos_shift = test_option['pos_shift']
neg_shift = test_option['neg_shift']
use_pos_pos_trials = test_option['use_pos_pos_trials']
count_top1_keyword = test_option['count_top1_keyword']
count_best_chunk = test_option['count_best_chunk']

os.makedirs(test_result_dir, exist_ok=True)

def backup_configs():
     # train config backup in result dir
     train_f = open("{}/train.yaml".format(test_result_dir), 'w')
     yaml.dump(train_config, train_f)
     # test config backup in result dir
     test_f = open("{}/test.yaml".format(test_result_dir), 'w')
     yaml.dump(test_config, test_f)

def make_fenyinta():
    positive_scp = make_dict_from_file(testset_dir+'/positive.wav.scp')
    negative_scp = make_dict_from_file(testset_dir+'/negative.wav.scp')
    positive2target = make_dict_from_file(testset_dir+'/positive.txt')
    keyword2phn = make_dict_from_file(testset_dir+'/keyword2phnseq')
    phn2id = make_dict_from_file(train_data_dir+'/phone2id.txt')

    keyword2phnid = {k: [phn2id[p] for p in v.split(" ")] for k, v in keyword2phn.items()}
    keyword2id = {k: i for i, k in enumerate(keyword2phnid.keys())}
    save_material = {
        'keyword2phn': keyword2phn,
        'keyword2phnid': keyword2phnid
    }
    meta_path = os.path.join(test_result_dir, 'fenyinta_meta.pt')
    torch.save(save_material, meta_path)
    positive_list_writer = open(os.path.join(test_result_dir, 'positive.datalist.txt'), 'w')
    for utt, wav in positive_scp.items():
        one_obj = dict(
            key=utt,
            sph=wav,
            target_keyword_id=keyword2id[positive2target[utt]],
            target_keyword = positive2target[utt]
        )
        positive_list_writer.write("{}\n".format(json.dumps(one_obj)))
    
    negative_list_writer = open(os.path.join(test_result_dir, 'negative.datalist.txt'), 'w')
    for utt, wav in negative_scp.items():
        one_obj = dict(
            key=utt,
            sph=wav
        )
        negative_list_writer.write("{}\n".format(json.dumps(one_obj)))


FBANK_DEFAULT_SETTING = {
    'num_mel_bins': 40, 'frame_length': 25, 'frame_shift': 10
}

def read_data_list(fname):
    dlist = []
    with open(fname) as tf:
        for line in tf.readlines():
            line = line.strip()
            mix_obj = json.loads(line)
            dlist.append(mix_obj)
    return dlist



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

def make_batch_keyword(keyword2id):

    keyword_list = []
    keyword_len = []
    idx2keyword = {}
    keyword2idx = {}
    for i, (keyword, phn_seq) in enumerate(keyword2id.items()):
        idx2keyword[i] = keyword
        keyword2idx[keyword] = i
        phn_seq = [1] + phn_seq + [2]
        phn_seq = torch.tensor(phn_seq).to(torch.long)
        keyword_list.append(phn_seq)
        keyword_len.append(len(phn_seq))
    keyword_list = pad_sequence(keyword_list, batch_first=True, padding_value=0)
    return (keyword_list, torch.tensor(keyword_len), idx2keyword, keyword2idx)


def print_feat(feat):
    with open("test_feat_print.txt", 'a') as f_feat:
        for i, keyword_feat in enumerate(feat):
            for j, frame_feat in enumerate(keyword_feat):
                f_feat.write("\n{}: {}: [".format(i, j))
                for r, f in enumerate(frame_feat):
                    if r == 0:
                        f_feat.write("{}".format(f))
                    else:
                        f_feat.write(", {}".format(f))
                f_feat.write("]")


def inference(model, test_list, keyword_material, idx2keyword, keyword2idx):
    keyword, keyword_len = keyword_material
    num_keyword = keyword.size(0)
    result = {}
    utt2keyword_idx = {}
    for i, obj in enumerate(test_list):
        if i % 300 == 0:
            print (i)
        key = obj['key']
        if 'target_keyword' in obj:
            utt2keyword_idx[key] = keyword2idx[obj['target_keyword']]
        wav = torchaudio.load(obj['sph'])[0]
        fbank = kaldi.fbank(wav, **FBANK_DEFAULT_SETTING)
        fbank = fbank.unsqueeze(0)
        fbank = fbank.repeat(num_keyword, 1, 1)
        fbank_len = torch.tensor([fbank.size(1) for x in range(num_keyword)])
        input_data = (fbank, fbank_len, keyword, keyword_len)
        input_data = (d.to('cuda:0') for d in input_data)
        det_result, hyp = model.evaluate(input_data)

        max_idx = torch.argmax(det_result).item()
        hyp = hyp[max_idx].tolist()
        hyp = [h for h in hyp if h != 0]
        hyp = [id2word[h] for h in hyp if h in id2word]
        det_result = det_result.view(-1).to('cpu').tolist()
        result.update({key: det_result})
    return result, utt2keyword_idx


def inference_shift(model, test_list, keyword_material, idx2keyword, keyword2idx, shift=False, chunk_size=chunk_size):
    keyword, keyword_len = keyword_material
    num_keyword = keyword.size(0)
    result = {}
    utt2keyword_idx = {}
    for i, obj in enumerate(test_list):
        if i < 50:
            print("utt: {}".format(i))
        elif i % 300 == 0:
            print("utt: {}".format(i))
        key = obj['key']
        if 'target_keyword' in obj:
            utt2keyword_idx[key] = keyword2idx[obj['target_keyword']]
        wav = torchaudio.load(obj['sph'])[0]
        fbank = kaldi.fbank(wav, **FBANK_DEFAULT_SETTING)
        fbank = fbank.unsqueeze(0)
        fbank = fbank.repeat(num_keyword, 1, 1)
        fbank_len = torch.tensor([fbank.size(1) for x in range(num_keyword)])
        j = 0
        result[key] = []
        if not shift:
            chunk_size = fbank.size(1)
        while j * chunk_shift + chunk_size <= fbank.size(1):
            if  j >= 1000 and j % 1000 == 0:
                print("chunk: {}".format(j))
            chunk = fbank[:, j * chunk_shift:j * chunk_shift + chunk_size, :]
            chunk_len = torch.tensor([chunk_size for x in range(num_keyword)])
            #print_feat(chunk)
            input_data = (chunk, chunk_len, keyword, keyword_len)
            input_data = (d.to('cuda:0') for d in input_data)
            det_result, hyp = model.evaluate(input_data)

            max_idx = torch.argmax(det_result).item()
            det_result = det_result.view(-1).to('cpu').tolist()
            result[key].append(det_result)
            j += 1
    return result, utt2keyword_idx
         

def plot_roc(positive_result, utt2keyword_idx, negative_result):
    pos_hyp = []
    pos_ref = []
    for utt, result in positive_result.items():
        pos_hyp.extend(result)
        target_keyword_idx = utt2keyword_idx[utt]
        one_hot = [0 for x in range(len(result))]
        one_hot[target_keyword_idx] = 1
        pos_ref.extend(one_hot)
    
    neg_hyp = []    
    neg_ref = []
    for utt, result in negative_result.items():
        neg_hyp.extend(result)
        one_hot = [0 for x in range(len(result))]
        neg_ref.extend(one_hot)


    fpr, tpr, thresholds = roc_curve(pos_ref+neg_ref, pos_hyp+neg_hyp)

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig('unet.transformer.png', dpi=400)


def plot_roc_shift(positive_result, utt2keyword_idx, negative_result, test_result_dir, analysis_id):
    pos_hyp = []
    pos_ref = []
    for utt, result in positive_result.items():
        target_keyword_idx = utt2keyword_idx[utt]
        utt_pos_hyp = []
        utt_pos_ref = []
        for chunk in result:
            if use_pos_pos_trials:
                utt_pos_hyp.extend(chunk)
                target_keyword_idx = utt2keyword_idx[utt]
                one_hot = [0 for x in range(len(chunk))]
                one_hot[target_keyword_idx] = 1
                utt_pos_ref.extend(one_hot)
            else:
                chunk_target_score = chunk[target_keyword_idx]
                if count_top1_keyword:
                    if chunk_target_score != max(chunk):
                        chunk_target_score = 0
                utt_pos_hyp.append(chunk_target_score)
                utt_pos_ref.append(1)
        if use_pos_pos_trials:
            pos_hyp.extend(utt_pos_hyp)
            pos_ref.extend(utt_pos_ref)
        else:
            if count_best_chunk:
                pos_hyp.append(max(utt_pos_hyp))
                pos_ref.append(max(utt_pos_ref))
            else:
                pos_hyp.extend(utt_pos_hyp)
                pos_ref.extend(utt_pos_ref)

    neg_hyp = []    
    neg_ref = []
    for utt, result in negative_result.items():
        for chunk in result:
            if count_top1_keyword:
                chunk_target_score = max(chunk)
                neg_hyp.append(chunk_target_score)
                neg_ref.append(0)
            else:
                neg_hyp.extend(chunk)
                one_hot = [0 for x in range(len(chunk))]
                neg_ref.extend(one_hot)

    fpr, tpr, thresholds = roc_curve(pos_ref+neg_ref, pos_hyp+neg_hyp)

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt_path = os.path.join(test_result_dir, 'unet.transformer_{}.png'.format(analysis_id))
    plt.savefig(plt_path, dpi=400)


def plot_roc_shift_one_word(positive_result, utt2keyword_idx, negative_result, test_result_dir, analysis_id, one_word, one_word_idx):
    pos_hyp = []
    pos_ref = []
    one_word_py = ''.join([ i[0] for i in pinyin(one_word, style=Style.FIRST_LETTER) ])
    for utt, result in positive_result.items():
        target_keyword_idx = utt2keyword_idx[utt]
        if target_keyword_idx != one_word_idx:
            continue
        utt_pos_hyp = []
        utt_pos_ref = []
        for chunk in result:
            chunk_target_score = chunk[target_keyword_idx]
            if count_top1_keyword:
                if chunk_target_score != max(chunk):
                    chunk_target_score = 0
            utt_pos_hyp.append(chunk_target_score)
            utt_pos_ref.append(1)
        if count_best_chunk:
            pos_hyp.append(max(utt_pos_hyp))
            pos_ref.append(max(utt_pos_ref))
        else:
            pos_hyp.extend(utt_pos_hyp)
            pos_ref.extend(utt_pos_ref)

    neg_hyp = []    
    neg_ref = []
    for utt, result in negative_result.items():
        for chunk in result:
            chunk_target_score = chunk[one_word_idx]
            if count_top1_keyword:
                if chunk_target_score != max(chunk):
                    chunk_target_score = 0
            neg_hyp.append(chunk_target_score)
            neg_ref.append(0)

    fpr, tpr, thresholds = roc_curve(pos_ref+neg_ref, pos_hyp+neg_hyp)

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for {}'.format(one_word_py))
    plt.legend(loc="lower right")
    plt_path = os.path.join(test_result_dir, 'unet.transformer_{}-{}.png'.format(analysis_id, one_word_idx))
    plt.savefig(plt_path, dpi=400)


if __name__ == '__main__':
    
    backup_configs()
    result_id = '_'.join([str(int(pos_shift)), str(int(neg_shift))])
    analysis_id = '_'.join([str(int(pos_shift)), str(int(neg_shift)), str(int(use_pos_pos_trials)), str(int(count_top1_keyword)), str(int(count_best_chunk))])
    result_path = os.path.join(test_result_dir, 'test.result_{}.pt'.format(result_id))
    meta_path = os.path.join(test_result_dir, 'fenyinta_meta.pt')
    if not os.path.exists(meta_path): 
        # 如果第一次做测试，需要将测试数据的一些信息存储在一个统一格式的文件fenyinta_meta.pt，方便后续测试
        # 直接使用
        make_fenyinta()
    test_material = torch.load(meta_path)
    model = load_model()
    keyword2id = {k: [int(vv) for vv in v] for k, v in test_material['keyword2phnid'].items()}
    keyword, keyword_len, idx2keyword, keyword2idx = make_batch_keyword(keyword2id)

    positive_list = read_data_list(os.path.join(test_result_dir, 'positive.datalist.txt'))
    positive_result_path = os.path.join(test_result_dir, 'test.result_pos_{}.pt'.format(int(pos_shift)))
    if os.path.exists(positive_result_path):
        test_result = torch.load(positive_result_path)
        positive_result = test_result['pos_hyp']
        utt2keyword_idx = test_result['pos_utt2keyword_idx']
    else:
        positive_result, utt2keyword_idx = inference_shift(model, positive_list, (keyword, keyword_len), idx2keyword, keyword2idx, shift=pos_shift)
        positive_test_result = {
            'pos_hyp': positive_result,
            'pos_utt2keyword_idx': utt2keyword_idx
        }
        torch.save(positive_test_result, positive_result_path)

    negative_list = read_data_list(os.path.join(test_result_dir, 'negative.datalist.txt'))
    negative_result_path = os.path.join(test_result_dir, 'test.result_neg_{}.pt'.format(int(neg_shift)))
    if os.path.exists(negative_result_path):
        test_result = torch.load(negative_result_path)
        negative_result = test_result['neg_hyp']
    else:
        negative_result, _ = inference_shift(model, negative_list, (keyword, keyword_len), idx2keyword, keyword2idx, shift=neg_shift)
        negative_test_result = {
            'neg_hyp': negative_result
        }
        torch.save(negative_test_result, negative_result_path)
    
    # 由于不同的阈值会产生不同的 Recall 和 FA，所以我这里将每一条语言在每个关键词上的分数都存在了test.result.pt 中，
    # 方便后续确定每个词的阈值。
    test_result = {
        'pos_hyp': positive_result, 'neg_hyp': negative_result, 'pos_utt2keyword_idx': utt2keyword_idx,
        'idx2keyword': idx2keyword
    }
    #最终会根据测试结果统一画一个 roc 曲线。
    plot_roc_shift(positive_result, utt2keyword_idx, negative_result, test_result_dir, analysis_id)
    if plot_each_word:
        for keyword, keywordidx in keyword2idx.items():
            print(keyword, keywordidx)
            plot_roc_shift_one_word(positive_result, utt2keyword_idx, negative_result, test_result_dir, analysis_id, keyword, keywordidx)
