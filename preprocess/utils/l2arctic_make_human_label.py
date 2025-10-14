#!/usr/bin/env python3

import re
import os
import json
from typing import Dict, List, Tuple, Optional
import sys
## input human label utt example, need to load text of each interval in iterm phones.
# File type = "ooTextFile"
# Object class = "TextGrid"

# xmin = 0 
# xmax = 3.851972789115646 
# tiers? <exists> 
# size = 3 
# item []: 
#     item [1]:
#         class = "IntervalTier" 
#         name = "words" 
#         xmin = 0 
#         xmax = 3.851972789115646 
#         intervals: size = 16 
#         intervals [1]:
#             xmin = 0 
#             xmax = 0.09 
#             text = "" 
#         intervals [2]:
#             xmin = 0.09 
#             xmax = 0.37 
#             text = "the" 
#         intervals [3]:
#             xmin = 0.37 
#             xmax = 0.81 
#             text = "ship" 
#         intervals [4]:
#             xmin = 0.81 
#             xmax = 0.9 
#             text = "" 
#         intervals [5]:
#             xmin = 0.9 
#             xmax = 1.35 
#             text = "should" 
#         intervals [6]:
#             xmin = 1.35 
#             xmax = 1.61 
#             text = "be" 
#         intervals [7]:
#             xmin = 1.61 
#             xmax = 1.84 
#             text = "in" 
#         intervals [8]:
#             xmin = 1.84 
#             xmax = 2.25 
#             text = "within" 
#         intervals [9]:
#             xmin = 2.25 
#             xmax = 2.28 
#             text = "" 
#         intervals [10]:
#             xmin = 2.28 
#             xmax = 2.43 
#             text = "a" 
#         intervals [11]:
#             xmin = 2.43 
#             xmax = 2.75 
#             text = "week" 
#         intervals [12]:
#             xmin = 2.75 
#             xmax = 2.88 
#             text = "or" 
#         intervals [13]:
#             xmin = 2.88 
#             xmax = 2.99 
#             text = "" 
#         intervals [14]:
#             xmin = 2.99 
#             xmax = 3.26 
#             text = "ten" 
#         intervals [15]:
#             xmin = 3.26 
#             xmax = 3.76 
#             text = "days" 
#         intervals [16]:
#             xmin = 3.76 
#             xmax = 3.851972789115646 
#             text = "" 
#     item [2]:
#         class = "IntervalTier" 
#         name = "phones" 
#         xmin = 0 
#         xmax = 3.851972789115646 
#         intervals: size = 35 
#         intervals [1]:
#             xmin = 0 
#             xmax = 0.09 
#             text = "sil" 
#         intervals [2]:
#             xmin = 0.09 
#             xmax = 0.16 
#             text = "DH" 
#         intervals [3]:
#             xmin = 0.16 
#             xmax = 0.37 
#             text = "AH0" 
#         intervals [4]:
#             xmin = 0.37 
#             xmax = 0.55 
#             text = "SH" 
#         intervals [5]:
#             xmin = 0.55 
#             xmax = 0.63 
#             text = "IH1" 
#         intervals [6]:
#             xmin = 0.63 
#             xmax = 0.81 
#             text = "P" 
#         intervals [7]:
#             xmin = 0.81 
#             xmax = 0.9 
#             text = "sp" 
#         intervals [8]:
#             xmin = 0.9 
#             xmax = 1.14 
#             text = "SH" 
#         intervals [9]:
#             xmin = 1.14 
#             xmax = 1.22 
#             text = "UH1" 
#         intervals [10]:
#             xmin = 1.22 
#             xmax = 1.35 
#             text = "D" 
#         intervals [11]:
#             xmin = 1.35 
#             xmax = 1.4 
#             text = "B" 
#         intervals [12]:
#             xmin = 1.4 
#             xmax = 1.61 
#             text = "IY0" 
#         intervals [13]:
#             xmin = 1.61 
#             xmax = 1.71 
#             text = "IH1" 
#         intervals [14]:
#             xmin = 1.71 
#             xmax = 1.84 
#             text = "N" 
#         intervals [15]:
#             xmin = 1.84 
#             xmax = 1.91 
#             text = "W" 
#         intervals [16]:
#             xmin = 1.91 
#             xmax = 1.97 
#             text = "IH0" 
#         intervals [17]:
#             xmin = 1.97 
#             xmax = 2.09 
#             text = "TH" 
#         intervals [18]:
#             xmin = 2.09 
#             xmax = 2.14 
#             text = "IH1" 
#         intervals [19]:
#             xmin = 2.14 
#             xmax = 2.25 
#             text = "N" 
#         intervals [20]:
#             xmin = 2.25 
#             xmax = 2.28 
#             text = "sp" 
#         intervals [21]:
#             xmin = 2.28 
#             xmax = 2.43 
#             text = "AH0" 
#         intervals [22]:
#             xmin = 2.43 
#             xmax = 2.51 
#             text = "W" 
#         intervals [23]:
#             xmin = 2.51 
#             xmax = 2.63 
#             text = "IY1" 
#         intervals [24]:
#             xmin = 2.63 
#             xmax = 2.75 
#             text = "K" 
#         intervals [25]:
#             xmin = 2.75 
#             xmax = 2.84 
#             text = "AO1" 
#         intervals [26]:
#             xmin = 2.84 
#             xmax = 2.88 
#             text = "R" 
#         intervals [27]:
#             xmin = 2.88 
#             xmax = 2.99 
#             text = "sp" 
#         intervals [28]:
#             xmin = 2.99 
#             xmax = 3.11 
#             text = "T" 
#         intervals [29]:
#             xmin = 3.11 
#             xmax = 3.17 
#             text = "EH1,IH,s" 
#         intervals [30]:
#             xmin = 3.17 
#             xmax = 3.26 
#             text = "N" 
#         intervals [31]:
#             xmin = 3.26 
#             xmax = 3.34 
#             text = "D" 
#         intervals [32]:
#             xmin = 3.34 
#             xmax = 3.45 
#             text = "EY1" 
#         intervals [33]:
#             xmin = 3.45 
#             xmax = 3.76 
#             text = "Z" 
#         intervals [34]:
#             xmin = 3.76 
#             xmax = 3.83 
#             text = "sp" 
#         intervals [35]:
#             xmin = 3.83 
#             xmax = 3.851972789115646 
#             text = "" 
#     item [3]:
#         class = "IntervalTier" 
#         name = "IPA" 
#         xmin = 0 
#         xmax = 3.851972789115646 
#         intervals: size = 3 
#         intervals [1]:
#             xmin = 0 
#             xmax = 3.11 
#             text = "" 
#         intervals [2]:
#             xmin = 3.11 
#             xmax = 3.17 
#             text = "ɛ,ɪ,s" 
#         intervals [3]:
#             xmin = 3.17 
#             xmax = 3.851972789115646 
#             text = "" 



    # 50     "c0860029": {
    # 51         "text": "肚子",
    # 52         "words": [
    # 53             {
    # 54                 "text": "肚子",
    # 55                 "phones": [
    # 56                     "d",
    # 57                     "u4",
    # 58                     "z",
    # 59                     "iy5"
    # 60                 ],
    # 61                 "phones-accuracy": [
    # 62                     1,
    # 63                     1,
    # 64                     1,
    # 65                     1
    # 66                 ]
    # 67             }
    # 68         ]
    # 69     },

def make_human_label_json(source_annot_dir: str, target_json_path: str, phone_list: List[str]):
    # pass
    target_label_dict = {}
    n_invalid_utt = 0
    n_utt = 0
    unk_phn_list = []
    for root, dirs, files in os.walk(source_annot_dir):
        # print(f"Processing {root}")
        if root.split('/')[-1] != 'annotation':
            continue
        subset = root.split('/')[-2]
        if subset not in ['NJS', 'TLV', 'TNI', 'TXHC', 'YKWK', 'ZHAA']:
            continue
        print(f"Processing {subset}")
        for file in files:
            if file.endswith('.TextGrid'):
                n_utt += 1
                utt_id = re.sub('.TextGrid', '', file)
                utt_id = re.sub('arctic_', '', utt_id)
                utt_id = f"{subset}_{utt_id}"
                utt_annot_path = os.path.join(root, file)
                annot_dict = utt_annot_to_label(utt_annot_path, phone_list)
                if annot_dict is None:
                    n_invalid_utt += 1
                    continue
                utt_label = annot_dict['label']
                unk_phns = annot_dict['unk_phns']
                for unk_phn in unk_phns:
                    if unk_phn not in unk_phn_list:
                        unk_phn_list.append(unk_phn)
                target_label_dict[utt_id] = utt_label
    print(f"num of invalid utts: {n_invalid_utt}/{n_utt}")
    print(f"num of unk phones: {len(unk_phn_list)}")
    print(f"first 10 unk phones: {unk_phn_list[:10]}")
    with open(target_json_path, 'w', encoding='utf-8') as f:
        json.dump(target_label_dict, f, ensure_ascii=False, indent=4)


def utt_annot_to_label(utt_annot_path: str, phone_list: List[str]) -> Dict[str, any]:
    annot_phn_list = get_phone_sequence(utt_annot_path)
    annot_word_list = get_word_sequence(utt_annot_path)
    annot_word_seq = ' '.join(annot_word_list)
    phones = []
    phones_accuracy = []
    unk_phns = []
    for phn in annot_phn_list:
        if ',' in phn:
            connonical_phn, actual_phn, error_type = re.sub(r'\s+', '', phn).split(',')
            if error_type not in ['s', 'd']:
                # print(f"Error type {error_type} not supported, only support s")
                return None

            connonical_phn = re.sub(r'[0-9`)]', '', connonical_phn).upper()
            if connonical_phn not in phone_list:
                return None
                if connonical_phn not in unk_phns:
                    unk_phns.append(connonical_phn)
            phones.append(connonical_phn)
            phones_accuracy.append(0)
        else:
            phn = re.sub(r'[0-9`)]', '', phn).strip().upper()
            if phn not in phone_list:
                return None
                if phn not in unk_phns:
                    unk_phns.append(phn)
            phones.append(phn)
            phones_accuracy.append(1)

    return {
        'label': {
            'text': annot_word_seq,
            'words': [
                {
                    'text': annot_word_seq,
                    'phones': phones,
                    'phones-accuracy': phones_accuracy
                }
            ]
        },
        'unk_phns': list(unk_phns)
    }


def parse_textgrid_file(file_path: str) -> Dict[str, List[Dict[str, any]]]:
    """
    解析TextGrid文件，提取每个tier中所有intervals的文本内容
    
    Args:
        file_path: TextGrid文件路径
        
    Returns:
        字典，键为tier名称，值为包含interval信息的列表
        每个interval包含: xmin, xmax, text
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取总的时间范围
    xmin_match = re.search(r'xmin = ([\d.]+)', content)
    xmax_match = re.search(r'xmax = ([\d.]+)', content)
    
    if not xmin_match or not xmax_match:
        raise ValueError("无法解析TextGrid文件的时间范围")
    
    total_xmin = float(xmin_match.group(1))
    total_xmax = float(xmax_match.group(1))
    
    # 提取tier数量
    size_match = re.search(r'size = (\d+)', content)
    if not size_match:
        raise ValueError("无法解析TextGrid文件的tier数量")
    
    tier_count = int(size_match.group(1))
    
    result = {}
    
    # 解析每个tier
    for i in range(1, tier_count + 1):
        tier_pattern = rf'item \[{i}\]:\s*class = "([^"]+)"\s*name = "([^"]+)"\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*intervals: size = (\d+)\s*(.*?)(?=item \[{i+1}\]|$)'
        tier_match = re.search(tier_pattern, content, re.DOTALL)
        
        if not tier_match:
            continue
            
        tier_class = tier_match.group(1)
        tier_name = tier_match.group(2)
        tier_xmin = float(tier_match.group(3))
        tier_xmax = float(tier_match.group(4))
        intervals_size = int(tier_match.group(5))
        intervals_content = tier_match.group(6)
        
        # 解析intervals
        intervals = []
        for j in range(1, intervals_size + 1):
            interval_pattern = rf'intervals \[{j}\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"'
            interval_match = re.search(interval_pattern, intervals_content)
            
            if interval_match:
                interval_xmin = float(interval_match.group(1))
                interval_xmax = float(interval_match.group(2))
                interval_text = interval_match.group(3)
                
                intervals.append({
                    'xmin': interval_xmin,
                    'xmax': interval_xmax,
                    'text': interval_text
                })
        
        result[tier_name] = intervals
    
    return result


def get_phone_intervals(file_path: str) -> List[Dict[str, any]]:
    """
    专门提取phones tier中的所有intervals
    
    Args:
        file_path: TextGrid文件路径
        
    Returns:
        包含所有phone intervals的列表
    """
    textgrid_data = parse_textgrid_file(file_path)
    
    if 'phones' not in textgrid_data:
        raise ValueError("TextGrid文件中未找到'phones' tier")
    
    return textgrid_data['phones']


def get_word_intervals(file_path: str) -> List[Dict[str, any]]:
    """
    专门提取words tier中的所有intervals
    
    Args:
        file_path: TextGrid文件路径
        
    Returns:
        包含所有word intervals的列表
    """
    textgrid_data = parse_textgrid_file(file_path)
    
    if 'words' not in textgrid_data:
        raise ValueError("TextGrid文件中未找到'words' tier")
    
    return textgrid_data['words']


def get_non_empty_intervals(intervals: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    过滤掉文本为空的intervals
    
    Args:
        intervals: interval列表
        
    Returns:
        过滤后的interval列表
    """
    return [interval for interval in intervals if interval['text'].strip() != '']


def get_phone_sequence(file_path: str, include_silence: bool = False) -> List[str]:
    """
    获取phone序列
    
    Args:
        file_path: TextGrid文件路径
        include_silence: 是否包含静音标记(sil, sp等)
        
    Returns:
        phone序列列表
    """
    phone_intervals = get_phone_intervals(file_path)
    
    if not include_silence:
        # 过滤掉静音和空格
        phone_intervals = [interval for interval in phone_intervals 
                          if interval['text'].strip() not in ['', 'sil', 'sp']]
    
    return [interval['text'] for interval in phone_intervals]


def get_word_sequence(file_path: str) -> List[str]:
    """
    获取word序列
    
    Args:
        file_path: TextGrid文件路径
        
    Returns:
        word序列列表
    """
    word_intervals = get_word_intervals(file_path)
    word_intervals = get_non_empty_intervals(word_intervals)
    
    return [interval['text'] for interval in word_intervals]


def load_phone_list(phone_list_path: str) -> List[str]:
    with open(phone_list_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

# 示例用法
if __name__ == "__main__":
    # 假设有一个TextGrid文件
    # file_path = "example.TextGrid"
    source_annot_dir = sys.argv[1]
    phone_list_path = sys.argv[2]
    target_json_path = sys.argv[3]
    
    # 解析整个文件
    # textgrid_data = parse_textgrid_file(file_path)
    # print("所有tiers:", list(textgrid_data.keys()))
    
    # 获取phone intervals
    # phone_intervals = get_phone_intervals(file_path)
    # print(f"Phone intervals数量: {len(phone_intervals)}")
    
    # 获取phone序列
    # phone_sequence = get_phone_sequence(file_path)
    # print("Phone序列:", phone_sequence)
    
    # 获取word序列
    # word_sequence = get_word_sequence(file_path)
    # print("Word序列:", word_sequence)
    phone_list = load_phone_list(phone_list_path)
    print(f"phone_list: {phone_list}")
    make_human_label_json(source_annot_dir, target_json_path, phone_list)
    
    pass

