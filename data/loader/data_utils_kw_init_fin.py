# ref: wenet processor.py
import re
import math
import random
import torch
import torchaudio
import json
import copy
import itertools
import inspect

import numpy as np

from torch.nn.utils.rnn import pad_sequence
from scipy.io import wavfile
from scipy import signal
from typing_extensions import List, Dict, Tuple, Iterator, Any, Optional


RE_PATTERN = {'space': re.compile(r" +"), 'dot': re.compile(r"\.")}

# random factory
RANDOM_FACTOR = {
    'beta':np.random.beta, 'uniform': np.random.uniform, 'int': np.random.randint, 'random': np.random.random
}

# Pre-Defined Special Token
TEXT_SPEC_TOKEN = {
    'sos': None, 'eos': None, 'sok': None, 'eok': None, 'unk': None, 'with_trans': None,
    'psok': None, 'peok': None, 'punk': None
}


#TODO: Max mix speech
def permuate_labels(labels: List[int], conject_token: List=None, max_mix_num: int=3):
    ids = [x for x in range(len(labels))]
    permuates = []
    max_n_permuate = math.factorial(max_mix_num)
    for one_idx in itertools.permutations(ids):
        if conject_token:
            one_per = [labels[x] + [conject_token] for x in one_idx]
            one_per[-1] = one_per[-1][:-1]
        else:
            one_per = [labels[x] for x in one_idx]
        permuates.append(one_per)

    if len(permuates) < max_n_permuate: # assume 3mix in data
        pad_per = permuates[0]
        for x in range(max_n_permuate-len(permuates)):
            permuates.append(pad_per)
    return permuates


# split str "0.1 0.2 0.3" to float [0.1 0.2 0.3] 
# str to int; int to sym; tensor to str
def sym2float(sym_list: List[int]):
    int_list = list(map(lambda x: float(x), sym_list.split(" ")))
    return int_list
def int2sym(int_list: List[int]):
    if not isinstance(int_list, list):
        int_list = [int_list]
    int_list = [str(x) for x in int_list]
    return int_list
def sym2int(sym_list: List[int]):
    int_list = list(map(lambda x: int(x), sym_list.split(" ")))
    return int_list
def tensor2str(t: torch.Tensor):
    if isinstance(t, torch.Tensor):
        t = t.numpy()
    t = list(t)
    t = list(map(lambda x: str(x), t))
    return t

# sample positive keyword from asr label
def sample_kw_from_label(label: List, segment: List, min_keyword_len: int=2, max_keyword_len: int=6)->Tuple[List, int]:
    #match_len = 0
    #while match_len == 0:
    #segment = set_length_range(segment, min_keyword_len, max_keyword_len)
    seg_len = len(segment)
    seg_pos = random.randint(0, seg_len-1)
    kw = segment[seg_pos]
    kw_len = len(kw)
    if kw_len >= min_keyword_len and kw_len <= max_keyword_len:
        match_len = 1
    kw_pos = 0
    for i in range(seg_pos):
        kw_pos += len(segment[i])
    return (kw, kw_pos)

# sample positive keyword from asr label
def sample_kw_using_whole_label_from_label(label: List, segment: List)->Tuple[List, int]: 
    kw_pos = 0

    return (label, kw_pos)

# sample positive keyword from asr label
def sample_kw_using_whole_label(label: List, segment: List, min_keyword_len: int=2, max_keyword_len: int=6)->Tuple[List, int]:
    #match_len = 0
    #while match_len == 0:
    #segment = set_length_range(segment, min_keyword_len, max_keyword_len)
    kw = [ k for word in segment for k in word ]
    #seg_len = len(segment)
    #seg_pos = random.randint(0, seg_len-1)
    #kw = segment[seg_pos]
    #kw_len = len(kw)
    #if kw_len >= min_keyword_len and kw_len <= max_keyword_len:
    #    match_len = 1
    kw_pos = 0
    #for i in range(seg_pos):
    #    kw_pos += len(segment[i])
    return (kw, kw_pos)

def substitution_neg(positive_keyword: List[int], aux_lexicon: Dict, negative_keyword: List[int]) -> List[int]:
    #print("substitution_neg")
    n_sub = 1
    lexicon_by_init = aux_lexicon.get('by_init')
    lexicon_by_final = aux_lexicon.get('by_final')
    if len(positive_keyword) > 3:
        n_sub = random.randint(2, len(positive_keyword)-1)
    positive_idx = [x for x in range(len(positive_keyword))] 
    sub_idx = random.sample(positive_idx, k=n_sub)
    keyword = []
    for x, one_word in enumerate(positive_keyword):
        if x not in sub_idx:
            keyword.extend(unfold_list(one_word))
        elif len(one_word) != 2:
            keyword.extend(unfold_list(one_word))
            # keyword.append(one_word)
        else:
            pos_init = one_word[0]
            pos_final = one_word[1]
            is_sub_init = random.randint(0,1)
            if is_sub_init:
                try:
                    candidate_init = [ i for i in lexicon_by_final[str(pos_final)] if i != pos_init ]
                    if len(candidate_init) < 2:
                        return positive_keyword
                except:
                    print("get init list by final error: init: {}, final: {}".format(pos_init, pos_final))
                    return positive_keyword
                try:
                    one_sub_init = random.sample(candidate_init, k=1)
                except:
                    print("init sample error: candidate_init: {}, lexicon_by_final[str(pos_final)]: {}, final: {}".format(candidate_init, lexicon_by_final[str(pos_final)], pos_final))
                    return positive_keyword
                one_sub_word = [one_sub_init[0], pos_final]
            else:
                try:
                    candidate_final = [ f for f in lexicon_by_init[str(pos_init)] if f != pos_final ]
                    if len(candidate_final) < 2:
                        return positive_keyword
                except:
                    print("get final list by init error: init: {}, final: {}".format(pos_init, pos_final))
                    return positive_keyword
                try:
                    one_sub_final = random.sample(candidate_final, k=1)
                except:
                    print("final sample error: candidate_final: {}, lexicon_by_init[str(pos_init)]: {}, init: {}".format(candidate_final, lexicon_by_init[str(pos_init)], pos_init))
                one_sub_word = [pos_init, one_sub_final[0]]
            keyword.append(unfold_list(one_sub_word))
    #print("sub neg positive_keyword: {}, sample_neg: {}".format(positive_keyword, keyword))
    return keyword


def substitution_neg_by_lex(positive_keyword: List[int], aux_lexicon) -> List[int]:
    n_sub = 1
    lexicon_by_init = aux_lexicon.get('by_init')
    lexicon_by_final = aux_lexicon.get('by_final')
    if len(positive_keyword) > 3:
        n_sub = random.randint(2, len(positive_keyword)-1)
    positive_idx = [x for x in range(len(positive_keyword))] 
    sub_idx = random.sample(positive_idx, k=n_sub)
    keyword = []
    positive_keyword_unfold = unfold_list(positive_keyword)
    positive_target = [ 1 for _ in range(len(positive_keyword_unfold)) ]
    target = []
    for x, one_word in enumerate(positive_keyword):
        if x not in sub_idx:
            keyword.extend(unfold_list(one_word))
            target.extend([1 for _ in range(len(unfold_list(one_word)))])
        elif len(one_word) != 2:
            keyword.extend(unfold_list(one_word))
            target.extend([1 for _ in range(len(unfold_list(one_word)))])
        else:
            pos_init = one_word[0]
            pos_final = one_word[1]
            is_sub_init = random.randint(0,1)
            if is_sub_init:
                try:
                    candidate_init = [ i for i in lexicon_by_final[str(pos_final)] if i != pos_init ]
                    if len(candidate_init) < 2:
                        return positive_keyword_unfold, positive_target
                except:
                    print("get init list by final error: init: {}, final: {}".format(pos_init, pos_final))
                    return positive_keyword_unfold, positive_target
                try:
                    one_sub_init = random.sample(candidate_init, k=1)
                except:
                    print("init sample error: candidate_init: {}, lexicon_by_final[str(pos_final)]: {}, final: {}".format(candidate_init, lexicon_by_final[str(pos_final)], pos_final))
                    return positive_keyword_unfold, positive_target
                one_sub_word = [one_sub_init[0], pos_final]
                one_sub_target = [0, 1]
            else:
                try:
                    candidate_final = [ f for f in lexicon_by_init[str(pos_init)] if f != pos_final ]
                    if len(candidate_final) < 2:
                        return positive_keyword_unfold, positive_target
                except:
                    print("get final list by init error: init: {}, final: {}".format(pos_init, pos_final))
                    return positive_keyword_unfold, positive_target
                try:
                    one_sub_final = random.sample(candidate_final, k=1)
                except:
                    print("final sample error: candidate_final: {}, lexicon_by_init[str(pos_init)]: {}, init: {}".format(candidate_final, lexicon_by_init[str(pos_init)], pos_init))
                one_sub_word = [pos_init, one_sub_final[0]]
                one_sub_target = [1, 0]
            keyword.extend(unfold_list(one_sub_word))
            target.extend(one_sub_target)
    # target = torch.tensor(target)
    return keyword, target

def substitution_neg_by_phone2id(positive_keyword: List[int], aux_lexicon: Dict, p_change: float = 0.5, is_skip_unk: bool = True) -> Tuple[List[int], List[int]]:
    phone2id = aux_lexicon.get('phone2id', {})
    if is_skip_unk:
        phone2id = {k: v for k, v in phone2id.items() if k != 'unk'}
    phone_ids = list(phone2id.values())
    keyword = []
    target = []
    for phone_id in positive_keyword:
        if phone_id not in phone_ids:
            keyword.append(phone_id)
            target.append(1)
            continue
        if random.random() >= p_change:
            keyword.append(phone_id)
            target.append(1)
        else:
            phone_ids_exclude = [p for p in phone_ids if p != phone_id]
            keyword.append(random.choice(phone_ids_exclude))
            target.append(0)
    return keyword, target

# transform positive keyword to negative keyword by substitution according to given lexicon, under pinyin shengyun constraint, including:
# 1) change init or final to another legal one; 2) change one char to another char; 3) change tone only
def substitution_neg_by_lex_shengyun_constraint_tone(
    positive_keyword: List[int],
    aux_lexicon: Dict,
    p_change: float = 0.5,
    type_weights: Dict = None,
) -> Tuple[List[int], List[int]]:
    """
    使用说明
      - p_change: 每个“字”被修改的概率（0~1）
      - type_weights: 在“决定要改”后，四类方式的权重 {'init': w1, 'final': w2, 'both': w3, 'tone': w4}
    依赖的词典
      - aux_lexicon['by_init'][str(init)]   -> List[final]
      - aux_lexicon['by_final'][str(final)] -> List[init]
      - aux_lexicon['by_len']['1']          -> List[[init, final]] 仅用于 both（整字替换），不作为 init/final 的兜底
      - aux_lexicon['alt_tone'][final_id]   -> List[final_id] 只替换声调（同一韵母的其他声调ID列表，**不含自身**）
    返回
      - keyword_unfold: [init, final, init, final, ...]
      - phone_target  : 等长 0/1，1=未改，0=被改
    行为
      - 单音素（len!=2）保持不改（与原函数一致）
      - 若某类替换无候选，则按“优先级”回退到其他类；最终仍无候选则该字不改
    """
    by_init   = aux_lexicon.get('by_init', {})
    by_final  = aux_lexicon.get('by_final', {})
    legal_pairs = aux_lexicon.get('by_len', {}).get('1', None)  # 仅供 both 使用
    legal_pairs = [ p[0] for p in legal_pairs ]
    alt_tone_map = aux_lexicon.get('alt_tone', {})  # {final_id: [other_tone_final_ids]}

    # 归一化四类权重（tone 默认 0，不改变旧行为）
    tw = type_weights or {}
    w_i = float(tw.get('init',  0.5))
    w_f = float(tw.get('final', 0.5))
    w_b = float(tw.get('both',  0.0))
    w_t = float(tw.get('tone',  0.0))   # NEW
    s = w_i + w_f + w_b + w_t
    if s <= 0:
        w_i, w_f, w_b, w_t = 0.5, 0.5, 0.0, 0.0
        s = 1.0
    p_i, p_f, p_b, p_t = w_i / s, w_f / s, w_b / s, w_t / s

    def pick_type() -> str:
        r = random.random()
        if r < p_i:  return 'init'
        r -= p_i
        if r < p_f:  return 'final'
        r -= p_f
        if r < p_b:  return 'both'
        return 'tone'  # 剩余概率给 tone

    def try_change_init(cur_i: int, cur_f: int) -> Optional[Tuple[int,int]]:
        # 只改声母：同韵母的其他声母
        try:
            cand = [i for i in by_final[str(cur_f)] if i != cur_i]
        except Exception:
            cand = []
        if not cand:
            return None
        ni = random.choice(cand)
        return ni, cur_f

    def try_change_final(cur_i: int, cur_f: int) -> Optional[Tuple[int,int]]:
        # 只改韵母：同声母的其他韵母
        try:
            cand = [f for f in by_init[str(cur_i)] if f != cur_f]
        except Exception:
            cand = []
        if not cand:
            return None
        nf = random.choice(cand)
        return cur_i, nf

    def try_change_both(cur_i: int, cur_f: int) -> Optional[Tuple[int,int]]:
        # 改整字：从所有合法对里随机挑一个≠原字（不依赖 by_init/by_final）
        if not legal_pairs or len(legal_pairs) <= 1:
            return None
        for _ in range(5):
            ni, nf = random.choice(legal_pairs)
            if ni != cur_i or nf != cur_f:
                return ni, nf
        candidates = [p for p in legal_pairs if not (p[0] == cur_i and p[1] == cur_f)]
        if not candidates:
            return None
        ni, nf = random.choice(candidates)
        return ni, nf

    def try_change_tone(cur_i: int, cur_f: int) -> Optional[Tuple[int,int]]:
        # NEW: 只改声调（韵母ID在 alt_tone_map 中查同韵母其它声调）
        cand = None
        # 兼容 key 既可能是 int 也可能是 str
        if cur_f in alt_tone_map:
            cand = alt_tone_map[cur_f]
        elif str(cur_f) in alt_tone_map:
            cand = alt_tone_map[str(cur_f)]
        if not cand:
            return None
        nf = random.choice(cand)  # cand 已不含自身
        return cur_i, nf

    keyword: List[int] = []
    target:  List[int] = []

    for one_word in positive_keyword:
        # 非 [init, final] 结构：保持不改
        if not isinstance(one_word, list) or len(one_word) != 2:
            uw = unfold_list(one_word)
            keyword.extend(uw)
            target.extend([1]*len(uw))
            continue

        cur_i, cur_f = one_word[0], one_word[1]

        # 是否触发修改
        if random.random() >= float(p_change):
            keyword.extend([cur_i, cur_f])
            target.extend([1, 1])
            continue

        typ = pick_type()
        changed_pair: Optional[Tuple[int,int]] = None

        if typ == 'init':
            # 优先级：init -> final -> both -> tone
            changed_pair = (try_change_init(cur_i, cur_f)
                            or try_change_final(cur_i, cur_f)
                            or try_change_both(cur_i, cur_f)
                            or try_change_tone(cur_i, cur_f))
        elif typ == 'final':
            # 优先级：final -> init -> both -> tone
            changed_pair = (try_change_final(cur_i, cur_f)
                            or try_change_init(cur_i, cur_f)
                            or try_change_both(cur_i, cur_f)
                            or try_change_tone(cur_i, cur_f))
        elif typ == 'both':
            # 优先级：both -> init -> final -> tone
            changed_pair = (try_change_both(cur_i, cur_f)
                            or try_change_init(cur_i, cur_f)
                            or try_change_final(cur_i, cur_f)
                            or try_change_tone(cur_i, cur_f))
        else:  # 'tone'
            # 优先级：tone -> final -> init -> both
            changed_pair = (try_change_tone(cur_i, cur_f)
                            or try_change_final(cur_i, cur_f)
                            or try_change_init(cur_i, cur_f)
                            or try_change_both(cur_i, cur_f))

        if changed_pair is None:
            # 没有任何可替换候选：该字不改
            keyword.extend([cur_i, cur_f])
            target.extend([1, 1])
        else:
            ni, nf = changed_pair
            keyword.extend([ni, nf])
            # 标注 phone 级 target：0=被改，1=未改
            target.extend([int(ni == cur_i), int(nf == cur_f)])

    return keyword, target

# transform positive keyword to negative keyword by substitution according to given lexicon, under pinyin shengyun constraint, including: 1. change init or final to another legal one; 2. change one char to another char
def substitution_neg_by_lex_shengyun_constraint(
    positive_keyword: List[int],
    aux_lexicon: Dict,
    p_change: float = 0.5,
    type_weights: Dict = None,
) -> Tuple[List[int], List[int]]:
    """
    使用说明
      - p_change: 每个“字”被修改的概率（0~1）
      - type_weights: 在“决定要改”后，三类方式的权重 {'init': w1, 'final': w2, 'both': w3}
    依赖的词典
      - aux_lexicon['by_init'][str(init)]  -> List[final]
      - aux_lexicon['by_final'][str(final)] -> List[init]
      - aux_lexicon['by_len']['1'] -> List[[init, final]] 仅用于 both（整字替换），不作为 init/final 的兜底
    返回
      - keyword_unfold: [init, final, init, final, ...]
      - phone_target  : 等长 0/1，1=未改，0=被改
    行为
      - 单音素（len!=2）保持不改（与原函数一致）
      - 若某类替换无候选，则按“优先级”回退到其他类；最终仍无候选则该字不改
    """
    by_init  = aux_lexicon.get('by_init', {})
    by_final = aux_lexicon.get('by_final', {})
    legal_pairs = aux_lexicon.get('by_len', {}).get('1', None)  # 仅供 both 使用，可能为 None
    legal_pairs = [ p[0] for p in legal_pairs ]

    # 归一化三类权重
    tw = type_weights or {}
    w_i = float(tw.get('init',  0.5))
    w_f = float(tw.get('final', 0.5))
    w_b = float(tw.get('both',  0.0))
    s = w_i + w_f + w_b
    if s <= 0:
        w_i, w_f, w_b = 0.5, 0.5, 0.0
        s = 1.0
    p_i, p_f, p_b = w_i / s, w_f / s, w_b / s

    def pick_type() -> str:
        r = random.random()
        if r < p_i:  return 'init'
        r -= p_i
        if r < p_f:  return 'final'
        return 'both'

    def try_change_init(cur_i: int, cur_f: int) -> Optional[Tuple[int,int]]:
        # 只改声母：同韵母的其他声母
        try:
            cand = [i for i in by_final[str(cur_f)] if i != cur_i]
        except Exception:
            cand = []
        if not cand:
            return None
        ni = random.choice(cand)
        return ni, cur_f

    def try_change_final(cur_i: int, cur_f: int) -> Optional[Tuple[int,int]]:
        # 只改韵母：同声母的其他韵母
        try:
            cand = [f for f in by_init[str(cur_i)] if f != cur_f]
        except Exception:
            cand = []
        if not cand:
            return None
        nf = random.choice(cand)
        return cur_i, nf

    def try_change_both(cur_i: int, cur_f: int) -> Optional[Tuple[int,int]]:
        # 改整字：从所有合法对里随机挑一个≠原字（不依赖 by_init/by_final）
        if not legal_pairs or len(legal_pairs) <= 1:
            return None
        # 尝试几次随机命中不同于原字的 pair
        for _ in range(5):
            ni, nf = random.choice(legal_pairs)
            if ni != cur_i or nf != cur_f:
                return ni, nf
        # 退化为一次过滤后再选
        candidates = [p for p in legal_pairs if not (p[0] == cur_i and p[1] == cur_f)]
        if not candidates:
            return None
        ni, nf = random.choice(candidates)
        return ni, nf

    keyword: List[int] = []
    target:  List[int] = []

    for one_word in positive_keyword:
        # 非 [init, final] 结构：保持不改
        if not isinstance(one_word, list) or len(one_word) != 2:
            uw = unfold_list(one_word)
            keyword.extend(uw)
            target.extend([1]*len(uw))
            continue

        cur_i, cur_f = one_word[0], one_word[1]

        # 是否触发修改
        if random.random() >= float(p_change):
            keyword.extend([cur_i, cur_f])
            target.extend([1, 1])
            continue

        typ = pick_type()
        changed_pair: Optional[Tuple[int,int]] = None

        if typ == 'init':
            # 优先级：init -> final -> both
            changed_pair = try_change_init(cur_i, cur_f) \
                           or try_change_final(cur_i, cur_f) \
                           or try_change_both(cur_i, cur_f)
        elif typ == 'final':
            # 优先级：final -> init -> both
            changed_pair = try_change_final(cur_i, cur_f) \
                           or try_change_init(cur_i, cur_f) \
                           or try_change_both(cur_i, cur_f)
        else:
            # 优先级：both -> init -> final
            changed_pair = try_change_both(cur_i, cur_f) \
                           or try_change_init(cur_i, cur_f) \
                           or try_change_final(cur_i, cur_f)

        if changed_pair is None:
            # 没有任何可替换候选：该字不改
            keyword.extend([cur_i, cur_f])
            target.extend([1, 1])
        else:
            ni, nf = changed_pair
            keyword.extend([ni, nf])
            # 标注 phone 级 target：0=被改，1=未改
            target.extend([int(ni == cur_i), int(nf == cur_f)])

    return keyword, target

def substitution_neg_md(positive_keyword: List[int], aux_lexicon: Dict, max_sub_ratio: float=0.5, min_sub: int=1) -> List[int]:
    char_phones = aux_lexicon.get('by_len')['1']

    len_positive_phones = len(positive_keyword)
    if len_positive_phones == 1:
        n_sub = 1
    else:
        max_sub = int(max_sub_ratio * len_positive_phones)
        try:
            assert max_sub >= min_sub, "max_sub must be not less than min_sub"
        except AssertionError as e:
            print(f"Assertion failed: {e}, max_sub: {max_sub}, min_sub: {min_sub}, positive_keywords: {positive_keyword}, max_sub_ratio: {max_sub_ratio}")
            return positive_keyword, torch.tensor([1]*len(positive_keyword))
        n_sub = random.randint(min_sub, max_sub)
    sub_idx = random.sample(range(len_positive_phones), k=n_sub)

    keyword = []
    target = []
    for x, one_phone in enumerate(positive_keyword):

        if x not in sub_idx:
            # keyword.extend(unfold_list(one_word))
            keyword.append(one_phone)
            target.append(1)
        else:
            sub_char = random.choice(char_phones)
            sub_phone = random.choice(sub_char[0])
            keyword.append(sub_phone)
            target.append(0)
    target = torch.tensor(target)
    # print("substitution_neg_md() -> keyword: {}".format(keyword))
    # print("substitution_neg_md() -> target: {}".format(target))
    return keyword, target

def deletion_neg(positive_keyword: List[int], aux_lexicon: Dict, negative_keyword: List[int]) -> List[int]:
    #print("deletion_neg")
    if len(positive_keyword) < 3:
        return positive_keyword
    n_del = 1
    if len(positive_keyword) > 3:
        n_del = 2
    positive_idx = [x for x in range(1, len(positive_keyword)-1)] 
    del_idx = random.sample(positive_idx, k=n_del)
    keyword = [positive_keyword[x]  for x in range(len(positive_keyword)) if x not in del_idx]
    keyword = unfold_list(keyword)
    #print("del neg positive_keyword: {}, sample_neg: {}".format(positive_keyword, keyword))
    return keyword

def insertion_neg(positive_keyword: List[int], aux_lexicon: Dict, negative_keyword: List[int]) -> List[int]:
    #print("insertion_neg")
    char_phones = aux_lexicon.get('by_len')['1']
    n_insert = 1
    if len(positive_keyword) > 3:
        n_insert = 2
    positive_idx = [x for x in range(len(positive_keyword))] 
    insert_idx = random.sample(positive_idx, k=n_insert)
    keyword = []
    for i, k in enumerate(positive_keyword):
        keyword.append(k)
        if i in insert_idx:
            keyword.append(random.choice(char_phones))
    #print("ins neg positive_keyword: {}, sample_neg: {}".format(positive_keyword, keyword))
    return keyword

def shuffle_neg(positive_keyword: List[int], aux_lexicon: Dict, negative_keyword: List[int]) -> List[int]:
    #print("shuffle_neg")
    keyword = positive_keyword[:]
    init_seq = []
    final_seq = []
    for char in positive_keyword:
        assert len(char) <= 2
        if len(char) == 2:
            init_seq.append(char[0])
            final_seq.append(char[1])
        else:
            init_seq.append(None)
            final_seq.append(char[0])
    shuffle_times = 0 
    while keyword == positive_keyword:
        shuffle_times += 1
        random.shuffle(init_seq)
        random.shuffle(final_seq)
        keyword = [[init_seq[i], final_seq[i]] if init_seq[i] != None else [final_seq[i]] for i in range(len(init_seq))]
        # random.shuffle(keyword)
        if shuffle_times > 10:
            break
    if shuffle_times > 10:
        keyword = negative_keyword[:]
    #print("shuf neg positive_keyword: {}, sample_neg: {}".format(positive_keyword, keyword))
    return keyword

def full_neg(positive_keyword: List[int], aux_lexicon: Dict, negative_keyword: List[int]) -> List[int]:
    return negative_keyword

NEG_FAMILY = {0: substitution_neg, 1: deletion_neg, 2: insertion_neg, 3: shuffle_neg, 4: full_neg}

# --- YAML-driven pickers for initial keyword sampling and MD negative sampling ---

# 1) Initial keyword sampler (for sample_keyword)

# Map method names to functions
MD_SAMPLING_FUNCTIONS = {
    'sample_kw_from_label': sample_kw_from_label,
    'sample_kw_using_whole_label': sample_kw_using_whole_label,
    'sample_kw_using_whole_label_from_label': sample_kw_using_whole_label_from_label
}

# 2) MD negative sampling (for make_keyword_md's negative branch)

# Map method names to functions (both must return (keyword, phone_target))
MD_NEG_SAMPLING_FUNCTIONS = {
    'substitution_neg_by_lex': substitution_neg_by_lex,  # existing function
    'substitution_neg_by_lex_shengyun_constraint': substitution_neg_by_lex_shengyun_constraint,  # existing function
    'substitution_neg_by_lex_shengyun_constraint_tone': substitution_neg_by_lex_shengyun_constraint_tone,  # existing function
    'substitution_neg_md': substitution_neg_md,          # existing function
    'substitution_neg_by_phone2id': substitution_neg_by_phone2id,          # existing function
}

# Return the full chosen choice dict (e.g., {'name': 'xxx', 'prob': 0.7, 'params': {...}})
def _pick_choice_from_choices(spec: Dict) -> Optional[Dict]:
    try:
        if not spec:
            return None
        choices = spec.get('choices', [])
        if not choices:
            return None
        probs = [float(c.get('prob', 1.0)) for c in choices]
        total = sum(probs)
        if total <= 0:
            return None
        r = random.random()
        acc = 0.0
        for c, p in zip(choices, probs):
            acc += p / total
            if r <= acc:
                return c
        return choices[-1]
    except Exception:
        return None

# Filter a params dict by a callable's signature (only pass accepted kwargs)
def _filter_kwargs_by_signature(fn, params: Dict) -> Dict:
    if not isinstance(params, dict) or not params:
        return {}
    try:
        sig = inspect.signature(fn)
        return {k: v for k, v in params.items() if k in sig.parameters}
    except Exception:
        return {}

# --- end YAML-driven pickers ---

# save wav as PCM_S 16bit 16k: always use to test code
def save_wav(wav: torch.Tensor, names: str):
    if isinstance(names, list):
        names = "_".join(names)
    torchaudio.save(
        "{}.wav".format(names), wav, sample_rate=16000, encoding="PCM_S", bits_per_sample=16
    )

# Splice feats: append context 
def splice_feats(feats: torch.Tensor, left_context: int, right_context: int, seq=False) -> torch.Tensor:
    frames, nmel = feats.size()
    l_padding = torch.ones_like(torch.rand(left_context, nmel))
    r_padding = torch.ones_like(torch.rand(right_context, nmel))
    l_padding *= feats[0]
    r_padding *= feats[-1]
    feats = torch.cat([l_padding, feats, r_padding], dim=0)
    if seq:
        return feats
    else:
        splice_v = []
        for i in range(left_context + right_context + 1):
            v = feats[i:frames + i]
            splice_v.append(v)
        feats = torch.cat([v for v in splice_v], dim=-1)
        return feats

# max_energy: find the max energy frames: TODO: partially duplicated with time shifting 
def max_energy(wav: torch.Tensor, frame_length: int=400, hop_length: int=160, shift_type: str='mid') -> int:
    wav = wav.view(-1) # assume wav is singal channel speech [1, num_samples] multi channel is not supportTODO:
    num_frames = 1 + (len(wav) - frame_length) // hop_length
    energy = torch.zeros(num_frames, dtype=torch.float32)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = wav[start:end]
        energy[i] = torch.sum(frame ** 2)
    # max frame energy
    idx_frame = torch.argmax(energy) 
    dice_frame = random.randint(0,5)
    if dice_frame % 2 == 0:
        idx_frame = idx_frame - dice_frame
        idx_frame = idx_frame if idx_frame > 0 else 0
    else:
        idx_frame = idx_frame + dice_frame
    #idx_frame = idx_frame - dice_frame
    idx_sample = idx_frame * hop_length + frame_length 
    return idx_sample

# flatten list [[1,2,3],[4,5,6,[7]]] => [1,2,3,4,5,6,7]
def unfold_list(lst: List[int]) -> List[int]:
    if not isinstance(lst, list):
        lst = [lst]
    l = _unfold_list(lst)
    trans = int if re.search(RE_PATTERN['dot'], l) == None else float
    l = [trans(i) for i in l.split(" ") if i !=""]
    return l
# sub method of unfold_list 
def _unfold_list(lst: List[int]) -> List[int]:
    new = ""
    for x in lst:
        if isinstance(x, list):
            x = _unfold_list(x)
        x = str(x)
        new = new + x + " "
    return new

#concat tensor
def concat_tensor(
    data_list: List[torch.Tensor],
    seq_padding: bool=False,
    padding_value: int=0
) -> torch.Tensor:
    if seq_padding:
        tensor = pad_sequence(data_list, batch_first=True, padding_value=padding_value)
    else:
        tensor = torch.cat([x.unsqueeze(0) for x in data_list], dim=0)
    return tensor


# random one sample from a pool
def random_one(pools: List[int], pool_len: int) -> Tuple[List]:
    return (pools[random.randint(0, pool_len-1)])


# spec augmentation
def spec_augment(
        spec: torch.Tensor, config: Dict
    ) -> torch.Tensor:
    assert isinstance(spec, torch.Tensor)
    num_t_mask = config.get('num_t_mask', 2)
    num_f_mask = config.get('num_f_mask', 2)
    max_t = config.get('max_t', 20)
    max_f = config.get('max_f', 10)
    spec_prob = config.get('spec_prob', 0.5)
    aug_spec = spec.clone().detach()
    if random.uniform(0, 1) > spec_prob:
        return aug_spec
    else:
        max_frames = aug_spec.size(0)
        max_freq = aug_spec.size(1)
        # print("max_frames: {}, max_freq: {}".format(max_frames, max_freq))
        # time mask
        for i in range(num_t_mask):
            start = np.random.randint(0, max_frames - 1)
            length = np.random.randint(1, max_t)
            end = min(max_frames, start + length)
            aug_spec[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = np.random.randint(0, max_freq - 1)
            length = np.random.randint(1, max_f)
            end = min(max_freq, start + length)
            aug_spec[:, start:end] = 0
            # print("start: {}, end: {}".format(start, end))
        return aug_spec

# Speech augmentation: reverb, change speed
def wav_augment(
        waveform: torch.Tensor, config: Dict, sample_rate: int=16000
    ) -> torch.Tensor:
    # add reverb and change speech 
    # TODO: if change speech alignment and segment should change too!!!
    if config.get("volume", False):
        volume_sampler = config['volume']['sampler']
        volume_sampler_config = config['volume']['config']
        ratio = RANDOM_FACTOR[volume_sampler](**volume_sampler_config)
        waveform = ratio * waveform
    
    if config.get('speed_perturb', False):
        speed_factor = config.get('speed_perturb').get('factors', [0.9, 1.0, 1.1])
        factor = random.choice(speed_factor)
        if factor != 1.0:
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, [['tempo', str(factor)], ['rate', str(sample_rate)]]
            )
    return waveform


def reverb_aug(waveform: torch.Tensor, config: Dict, rirs: str=None) -> torch.Tensor:
    if len(waveform.size()) == 2:
        n = waveform.size(0)
    else:
        n = waveform.dim()
    try:
        assert n == 1, "number of waveform channel must be 1"
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        return None
    #try:
    #    #assert waveform.shape[1] >= 800, "waveform length must be larger than 800 samples"
    #    assert waveform.shape[1] >= 16000, "waveform length must be larger than 16000 samples"
    #except AssertionError as e:
    #    print(f"Assertion failed: {e}")
    #    print("waveform length: {}".format(waveform.shape[1]))
    #    return None
    if rirs:
        rirs_prob = config.get('rirs_prob', 0.4)
        if random.uniform(0,1) < rirs_prob:
            if isinstance(rirs, list):
                rirs_file = random.choice(rirs)
            _, rirs_src = wavfile.read(rirs_file)
            rirs_src = rirs_src / np.sqrt(np.sum(rirs_src**2))
            rirs_src = rirs_src.astype(np.float32)
            l = waveform.size(1)
            waveform = waveform[0].numpy()
            waveform = signal.convolve(waveform, rirs_src)[:l]
            waveform = torch.from_numpy(waveform)
            waveform = waveform.view(1,-1)
    return waveform

def add_noise(speech: torch.Tensor, noise: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    snr = random.randint(3, 10)
    signal_power = (speech**2).mean()
    noise_power = (noise**2).mean()

    target_noise_power = signal_power / (10**(snr / 10))

    scaling_factor = torch.sqrt(target_noise_power / noise_power)
    adjusted_noise = noise * scaling_factor
    return (speech + adjusted_noise, adjusted_noise)


def make_mix_wav(
        speech: torch.Tensor, ratios: List[float], noise: Optional[torch.Tensor] = None, 
        noise_ratio: Optional[List] = None, delay_prob: float=0.35, delay_type='random'
    )-> torch.Tensor:
    assert (delay_type in ['random', 'sequential'])
    if (random.random() < delay_prob) and (len(speech) > 1):
        for delay_idx in range(1, len(speech)): # apply delay from the second wavform
            if delay_idx == 1:
                delay_len = random.randint(0, 16000*3)
            else:
                if delay_type == 'sequential':
                    delay_len = delay_len + random.randint(2000, 16000*3)
                else:
                    delay_len = random.randint(0, 16000*3)
            zero_padding = torch.zeros(1, delay_len)
            speech[delay_idx] = torch.cat([zero_padding, speech[delay_idx]], dim=1)

    max_len = max([s.size(1) for s in speech])
    
    # speech = [padding_wav(s, max_len, 'zero') for s in speech] # padding_wav(target_wav, padding_len, padding_type)
    # noise = [padding_wav(n, max_len, 'repeat') for n in noise]
    speech = [
        pad for s in speech
        if (pad := padding_wav(s, max_len, 'zero')) is not None
    ]
    noise = [
        pad for n in noise
        if (pad := padding_wav(n, max_len, 'repeat')) is not None
    ]

    rms = [math.sqrt((s**2).mean()) for s in speech]
    rms = list(map(lambda x: x if x > 0.0001 else 1, rms)) # avoid devide very small value 0
    max_rms = max(rms)

    # print("speech length: {}".format(len(speech)))
    scaled_wav = [speech[i]*(max_rms/rms[i]) for i in range(len(speech))]
    # print("scaled_wav length: {}".format(len(scaled_wav)))
    # print("ratios length: {}".format(len(ratios)))
    scaled_wav = [scaled_wav[i]*ratios[i] for i in range(len(scaled_wav))]

    mix_wav = sum(scaled_wav)

    if len(noise) > 0:
        noise = sum(noise)
        mix_wav, noise = add_noise(mix_wav, noise) # Signal power is temporary set to SNR=1~10dB
        wav_group = scaled_wav + [noise]
    else:
        wav_group = scaled_wav[:]
    mix_wav = torch.clamp(mix_wav, -1.0, 1.0)

    return [mix_wav] + wav_group


# padding wav with zeros
def padding_wav(waveform: torch.Tensor, length: int, padding_type: str='zero')->torch.Tensor:
    assert (padding_type in ['zero', 'repeat'])
    assert (waveform.dim() == 2)
    try:
        assert waveform.size(0) == 1, "padding_wav() -> number of waveform channel must be 1"
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        return None
    # try:
    #     assert waveform.shape[1] >= 16000, "padding_wav() -> waveform length must be larger than 16000 samples"
    # except AssertionError as e:
    #     print(f"Assertion failed: {e}")
    #     return None
    wav_len = waveform.shape[1]
    if padding_type == 'repeat':
        if wav_len < length:
            repeat_time = length // wav_len + 1
            waveform = torch.tile(waveform.view(-1), (repeat_time,))
            waveform = waveform.view(1, -1)
        waveform = waveform[:,:length]
    else:
        assert (waveform.size(1) <= length)
        zeros = torch.zeros(1, length - wav_len)
        waveform = torch.cat([waveform, zeros], dim=1)
    return waveform

# trim wav by lenth and segment
def trim_wav(wav: torch.Tensor, segment: List[int]) -> torch.Tensor:
    head, tail = segment
    if head == -1:
        # when head = -1 tail is the target windows length
        head = 0
        wav_size = wav.size(1)
        res = wav_size - tail
        head = 0 if res <= 0 else int(random.uniform(0, res))
        tail = head + tail
    wav = wav[:,head: tail]
    return wav


# get segment idx by time stamp
def got_seg(v, segment):
    diff = float('inf')
    nearest_value, nearest_index = None, None
    for i, value in enumerate(segment):
        if abs(value - v) < diff:
            diff = abs(value - v)
            nearest_value = value
            nearest_index = i
    return nearest_value, nearest_index


# make corruption pairs, maybe triple or more
# detach_f: detach_functions, for self corruption the datalist will format as json object so detach_f will be json.loads
# for none target corruption detach_f is lambda x: x
def make_corrupt_party(
        corrupt_list: List[Any], corrupt_list_len: int, config: Dict, corruption_type: str='self'
    )-> Tuple[List, Dict, int]:
    assert (corruption_type in ['self', 'noise'])
    ratios = list()
    corruption_material = dict()
    n = config.get('num_corrupt')
    max_corrupt_num = config.get('num_corrupt')
    
    if config.get('random_num', False):
        assert(n > 1)
        n = RANDOM_FACTOR['int'](1, n+1)
    if config.get('prob', 1.2) < 1:
        dice = random.uniform(0,1)
        if dice > config.get('prob'):
            n = 0

    sampler = config.get('sampler')
    assert (sampler in RANDOM_FACTOR.keys())
    sampler_config = config.get('sampler_config')
    if corruption_type == 'self':
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)
    
    for i in (range(n)):
        one_corrupt = corrupt_list[np.random.randint(0, corrupt_list_len)]
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)
        if isinstance(one_corrupt, dict):
            corruption_material[i+1] = one_corrupt
        elif isinstance(one_corrupt, str):
            corruption_material[i+1] = {'sph': one_corrupt}
        else:
            raise NotImplementedError("corrupt list error")
    return (ratios, corruption_material, n, max_corrupt_num)


# time shiftting: shiftting keywords from waveforme to make sure that: when mix tow keywords they will completely overlaped
def time_shifting(
        wav: torch.Tensor, frame_length: int=400, hop_length: int= 160, shift_type: str='mid'
    )-> torch.Tensor:
    wav = wav.view(-1) # assume wav is singal channel speech [1, num_samples] multi channel is not supportTODO:
    num_frames = 1 + (len(wav) - frame_length) // hop_length
    energy = torch.zeros(num_frames, dtype=torch.float32)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = wav[start:end]
        energy[i] = torch.sum(frame ** 2)
    # max frame energy
    idx_frame = torch.argmax(energy) 
    idx_sample = idx_frame * hop_length + frame_length 
    if shift_type == 'mid':
        dest_pos = int(wav.size(0) / 2)
    shift = int(dest_pos-idx_sample)
    wav = torch.roll(wav, shift)
    return wav.view(1, -1) # NOTE: convert back to singal channel [1, num_sample]


# make segment: make segment head and tail => [0.5, 1.5] mains: wav will trimed from the 0.5s to 1.5s (1s) 
def make_segment(
        segments: List[int], win_len: int, trim_type: str='raw', dither: bool=False, sample_rate: int=16000
    ) -> Tuple[int]:
    seg_head_tail = []
    idx_head_tail = []
    for seg in segments:
        if seg == -1:
            seg_head_tail.append([-1, int(win_len*sample_rate)])
            idx_head_tail.append([-1, -1])
            continue
        if trim_type == 'raw':
            seg_head = seg[0]
            if dither:
                seg_head = seg_head - random.uniform(0, 0.1)
                seg_head = seg_head if seg_head > 0 else 0
            seg_tail = seg_head + win_len
            seg_head_tail.append([int(seg_head*sample_rate), int(seg_tail*sample_rate)])
            idx_head_tail.append([0, len(seg)])
        elif trim_type == 'completetrim': 
            # cut setence from one utterance; the sub-utterance will contain a complete content
            leng_wav = seg[-1][-1]
            if leng_wav > win_len:
                seg_head_range = leng_wav - win_len
                _, seg_idx = got_seg(seg_head_range, [s[0] for s in seg])
                seg_head_idx = seg_idx if seg_idx <= 1 else np.random.randint(0, seg_idx-1)
                seg_head = seg[seg_head_idx][0]
                seg_tail = seg_head + win_len
                _, seg_tail_idx = got_seg(seg_tail, [s[1] for s in seg])
                seg_tail_idx = seg_tail_idx + 1 if seg_tail_idx < len(seg) - 1 else seg_tail_idx
                seg_tail = seg[seg_tail_idx][1]
            else:
                seg_head = 0
                seg_tail = seg[-1][-1]
                seg_head_idx = 0 
                seg_tail_idx = len(seg)
            seg_head_tail.append([int(seg_head*sample_rate), int(seg_tail*sample_rate)])
            idx_head_tail.append([seg_head_idx, seg_tail_idx])
        else:
            raise NotImplementedError("Only support raw")
    return (seg_head_tail, idx_head_tail)


# detach corruption
#def detach_corruption(material: Dict) -> Tuple[List, List, List, List, List, List, List]:
def detach_corruption(material: Dict) -> Tuple[List, List, List, List, List, List]:
    keywords = [] 
    phn_labels = []
    #segment_labels = []
    bpe_labels = []
    labels = []
    kw_candidates = []
    b_kw_candidates = []
    for i, (_idx, info) in enumerate(material.items()):
        if 'keyword' in info:
            keywords.append(info['keyword'])
        if 'label' in info:
            labels.append(info['label'])
        if 'phn_label' in info:
            phn_labels.append(info['phn_label'])
        #if 'segment_label' in info:
        #    segment_labels.append(info['segment_label'])
        if 'bpe_label' in info:
            bpe_labels.append(info['bpe_label'])
        if 'b_kw_candidate' in info:
            b_kw_candidates.append(info['b_kw_candidate'])
        if 'kw_candidate' in info:
            kw_candidates.append(info['kw_candidate'])
    return keywords, labels, phn_labels, bpe_labels, kw_candidates, b_kw_candidates
    #return keywords, labels, phn_labels, segment_labels, bpe_labels, kw_candidates, b_kw_candidates


# insert special token in label sequence such as SOS: 0(start of sentence) 
# 1 2 3 4 5 -> "0" 1 2 3 4 5
def inject_special_token_md(
        keyword: List[int], keyword_length: int, label: List=None, 
        positive: bool=True, keyword_pos: int=None, special_token: Dict={}, bpe_label: List=None, 
        bpe_candidate: List=None
    )->Tuple[List, List, List, int, List]:
    org_keyword_len = len(keyword)
    TEXT_SPEC_TOKEN.update(special_token)
    new_phn_label = copy.deepcopy(label)
    new_bpe_label = copy.deepcopy(bpe_label) if bpe_label else [0]
    new_keyword = copy.deepcopy(keyword)
    kw_spec_mask = [0] * org_keyword_len

    if TEXT_SPEC_TOKEN['sos'] != None: # start of sentence
        new_phn_label = [TEXT_SPEC_TOKEN['sos']] + new_phn_label
        keyword_pos = keyword_pos + 1  if keyword_pos != None else keyword_pos # one token insert before the keyword
        
    if TEXT_SPEC_TOKEN['eos'] != None: # end of sentence
        new_phn_label = new_phn_label + [TEXT_SPEC_TOKEN['eos']] 
    
    if TEXT_SPEC_TOKEN['sop'] != None: # start of phone
        for i in range(org_keyword_len):
            new_keyword.insert(i*2, TEXT_SPEC_TOKEN['sop'])
            kw_spec_mask.insert(i*2, 1)

    if TEXT_SPEC_TOKEN['psok'] != None: # start of keyword
        new_keyword.insert(0, [TEXT_SPEC_TOKEN['psok']])
        kw_spec_mask.insert(0, 1)

    if TEXT_SPEC_TOKEN['peok'] != None: # end of keyword
        new_keyword.insert(len(new_keyword), [TEXT_SPEC_TOKEN['peok']])
        kw_spec_mask.insert(len(new_keyword), 1)

    if (TEXT_SPEC_TOKEN['with_trans']) and (positive): # modify keyword in label
        new_phn_label[keyword_pos: keyword_pos+keyword_length] = new_keyword
        if bpe_label:
            bpe_kw_head = bpe_candidate[keyword_pos]
            bpe_kw_tail = bpe_candidate[keyword_pos + keyword_length - 1] # max index = offset + length - 1
            bpe_kw = bpe_label[bpe_kw_head: bpe_kw_tail + 1] # keyword in bpe label
            bpe_kw.insert(0, [TEXT_SPEC_TOKEN['sok']])
            bpe_kw.insert(len(bpe_kw), [TEXT_SPEC_TOKEN['eok']])
            new_bpe_label[bpe_kw_head: bpe_kw_tail + 1] = bpe_kw

    return (new_keyword, new_phn_label, new_bpe_label, keyword_pos, kw_spec_mask)

# snipe_edges for waveform
def snipe_edge(waveform: torch.Tensor, hop_length: int=160):
    num_samples = waveform.size(1)
    edges = num_samples % hop_length
    return waveform[:,0:num_samples-edges]

def sample_keyword(candidate_seq: List[Any], segment_seq: List[Any], 
                    sample_func_choice: Dict=None) -> Tuple[List, int]:
    # Pick a function via YAML choices (with optional per-function params)
    choice = _pick_choice_from_choices(sample_func_choice)
    func_name = choice.get('name') if choice else None
    func_params = choice.get('params', {}) if choice else {}
    fn = MD_SAMPLING_FUNCTIONS.get(func_name, sample_kw_from_label)
    func_kwargs = _filter_kwargs_by_signature(fn, func_params)

    return fn(candidate_seq, segment_seq, **func_kwargs)

def make_keyword_md(
        candidate_seq: List[Any], segment_seq: List[Any],
        positive_prob: float, aux_lexicon: Dict=None, 
        sample_func_choice: Dict=None, neg_sample_func_choice: Dict=None, target_level: List=None
    ) -> Tuple[List, int, int, bool, List]:

    keyword, keyword_pos = sample_keyword(candidate_seq, segment_seq, sample_func_choice)
    # keyword_unfold = unfold_list(keyword)
    pos = True
    # target = torch.tensor([1]*len(keyword_unfold))
    # print("make_keyword_md() -> keyword len: {}, keyword_unfold len: {}, target len: {}".format(len(keyword), len(keyword_unfold), len(target)))

    dice = random.uniform(0,1)
    if dice > positive_prob: # negtivae sample
        target = []
        # choose MD negative function via YAML config; default to substitution_by_lex
        neg_choice = _pick_choice_from_choices(neg_sample_func_choice)
        neg_func_name = neg_choice.get('name') if neg_choice else None
        neg_func_params = neg_choice.get('params', {}) if neg_choice else {}
        md_neg_fn = MD_NEG_SAMPLING_FUNCTIONS.get(neg_func_name, substitution_neg_by_lex)
        neg_func_kwargs = _filter_kwargs_by_signature(md_neg_fn, neg_func_params)
        keyword, phone_target = md_neg_fn(keyword, aux_lexicon, **neg_func_kwargs)
        if 'phone' in target_level:
            target.extend(phone_target)
        if 'word' in target_level:
            # insert word-level target
            if 0 in target:
                target.insert(0, 0)
            else:
                target.insert(0, 1)
        target = torch.tensor(target)

        keyword_pos = -1
        pos = False
    else:
        keyword_ = keyword[:]
        keyword = unfold_list(keyword_)
        target = []
        if 'phone' in target_level:
            target.extend([1]*len(keyword))
        if 'word' in target_level:
            target.insert(0, 1)
        target = torch.tensor(target)
        # target = torch.tensor([1]*(len(keyword)+1))

    return (keyword, keyword_pos, len(keyword), pos, target)


# sample negative keyword from the whole corpus
def random_one_neg(neg_list: List[int], neg_len: int, pos_label: List, spk_id: str=None)->List[int]:
    neg = pos_label[0]
    flatten_label = unfold_list(pos_label)
    flatten_neg = unfold_list(neg)
    flatten_label = int2sym(flatten_label)
    flatten_neg = int2sym(flatten_neg)
    if spk_id != None:
        neg_spk = spk_id
    else:
        neg_spk = -1
    while (" ".join(flatten_neg) in " ".join(flatten_label)) or (neg_spk == spk_id):
        one_neg_list = neg_list[random.randint(0, neg_len-1)]
        one_neg_list = json.loads(one_neg_list)
        if spk_id != None:
            neg_spk = one_neg_list['key'].split('-')[0]
        one_neg_label = one_neg_list['phn_label']
        kw_candidate = one_neg_list.get('kw_candidate', None) 
        neg, _ = sample_kw_from_label(one_neg_label, kw_candidate)
        flatten_neg = unfold_list(neg)
        flatten_neg = int2sym(flatten_neg)
    return neg

def random_one_neg_from_lexicon(aux_lexicon: Dict, min_keyword_len: int, max_keyword_len: int, num_pre_sample: int, pos_label: List)->List[int]:
    neg = pos_label[0]
    lexicon_by_len = aux_lexicon['by_len']
    flatten_label = unfold_list(pos_label)
    flatten_neg = unfold_list(neg)
    flatten_label = int2sym(flatten_label)
    flatten_neg = int2sym(flatten_neg)
    pre_sample_lexicon = {}
    neg_keyword_len = random.randint(min_keyword_len, max_keyword_len)
    pre_sample_lexicon = random.sample(lexicon_by_len[str(neg_keyword_len)], num_pre_sample)
    while (" ".join(flatten_neg) in " ".join(flatten_label)):
        neg = random.choice(pre_sample_lexicon)
        flatten_neg = unfold_list(neg)
        flatten_neg = int2sym(flatten_neg)
    return neg
