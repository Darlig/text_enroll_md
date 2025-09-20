#!/usr/bin/env python3

import sys
import re
import json
import ast
import os
import subprocess

source_datalist = sys.argv[1]
character_timestamp = sys.argv[2]
segment_timestamp = sys.argv[3]
cut_datalist = sys.argv[4]
output_audio_root = sys.argv[5]  # 新增：分割后音频文件的根目录
min_second_dim_length = int(sys.argv[6]) if len(sys.argv) > 6 else 2  # 可配置的第二个维度长度阈值，默认为2
audio_extension_ms = int(sys.argv[7]) if len(sys.argv) > 7 else 0  # 音频拓展长度（毫秒），默认为0

#1001501_26b0ce87 [[270, 450], [450, 490], [490, 590], [590, 650], [650, 730], [730, 770], [770, 830], [830, 890], [890, 930], [930, 990], [990, 1030], [1030, 1150], [1150, 1465]]
#{"key": "Y0000003589_8WOJb2iiULs_S00222", "sph": "/work104/weiyang/data/wenetspeech/dataset/drama_samp500h/audio_cut3/audio/train/youtube_wav/B00014/Y0000003589_8WOJb2iiULs/Y0000003589_8WOJb2iiULs_S00222.wav", "bpe_label": [815, 47, 484, 13, 14, 1566, 754, 22, 1732, 1708, 765], "phn_label": [[64, 93], [58, 143], [32, 88], [20, 21], [22, 19], [7, 184], [10, 113], [3, 61], [44, 91], [72, 52], [64, 93]], "segment_label": [[[64, 93], [58, 143], [32, 88]], [[20, 21], [22, 19]], [[7, 184], [10, 113]], [[3, 61], [44, 91]], [[72, 52], [64, 93]]], "kw_candidate": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b_kw_candidate": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

datalist_dict = {}
with open(source_datalist) as f_datalist:
    for line in f_datalist:
        utt_datalist_dict = json.loads(line.strip())
        uttid = utt_datalist_dict["key"]
        datalist_dict[uttid] = utt_datalist_dict

c_timestamp_dict = {}
with open(character_timestamp) as f_timestamp:
    for line in f_timestamp:
        uttid, content = re.split(r'\s+', line.strip(), maxsplit=1)
        c_timestamp = ast.literal_eval(content)
        c_timestamp_dict[uttid] = c_timestamp

def get_segment_timestamp(datalist_dict, c_timestamp_dict):
    seg_timestamp_dict = {}
    for uttid, utt_datalist_dict in datalist_dict.items():
        if uttid in c_timestamp_dict:
            c_timestamp = c_timestamp_dict[uttid]
            segment_label = utt_datalist_dict["segment_label"]
            seg_len_list = [ len(seg) for seg in segment_label ]
            seg_timestamp_list = []
            i = 0
            for sl in seg_len_list:
                seg_ts = c_timestamp[i:i+sl]
                seg_timestamp_list.append(seg_ts)
                i += sl
            seg_timestamp_dict[uttid] = seg_timestamp_list
    return seg_timestamp_dict

def get_segment_time_range(seg_timestamp, extension_ms=0, audio_duration_ms=None):
    """
    从segment timestamp中获取起止时间（毫秒），并应用拓展
    seg_timestamp: [[start1, end1], [start2, end2], ...]
    extension_ms: 拓展长度（毫秒）
    audio_duration_ms: 音频总长度（毫秒），用于边界处理
    返回: (start_time_ms, end_time_ms)
    """
    if not seg_timestamp:
        return 0, 0
    
    print("seg_timestamp: {}".format(seg_timestamp))
    original_start_time = seg_timestamp[0][0][0]  # 第一个元素的第一个值
    original_end_time = seg_timestamp[-1][-1][1]   # 最后一个元素的第二个值
    
    # 应用拓展
    start_time = max(0, original_start_time - extension_ms)  # 不能小于0
    end_time = original_end_time + extension_ms
    
    # 如果提供了音频总长度，确保不超过音频边界
    if audio_duration_ms is not None:
        end_time = min(end_time, audio_duration_ms)
    
    return start_time, end_time

def get_audio_duration_ms(input_audio_path):
    """
    获取音频文件的时长（毫秒）
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', input_audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            duration_sec = float(result.stdout.strip())
            return int(duration_sec * 1000)  # 转换为毫秒
        else:
            print(f"Warning: Failed to get duration for {input_audio_path}")
            return None
    except Exception as e:
        print(f"Error getting duration for {input_audio_path}: {e}")
        return None

def cut_audio_file(input_audio_path, output_audio_path, start_ms, end_ms):
    """
    使用ffmpeg切分音频文件
    start_ms, end_ms: 起止时间（毫秒）
    """
    print("start_ms: ", start_ms, "end_ms: ", end_ms)
    try:
        # 将毫秒转换为秒
        start_sec = start_ms / 1000.0
        duration_sec = (end_ms - start_ms) / 1000.0
        
        # 使用ffmpeg切分音频
        cmd = [
            'ffmpeg', '-i', input_audio_path,
            '-ss', str(start_sec),
            '-t', str(duration_sec),
            '-acodec', 'copy',  # 直接复制音频流，不重新编码
            '-y',  # 覆盖输出文件
            output_audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Failed to cut audio {input_audio_path}: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error cutting audio {input_audio_path}: {e}")
        return False

def generate_output_audio_path(uttid, seg_idx, start_ms, end_ms, output_root):
    """
    生成输出音频文件路径，统一放入指定的根目录
    """
    # 创建基于uttid的子目录结构，避免文件名冲突
    # 使用uttid的前几个字符作为子目录
    subdir = uttid[:8] if len(uttid) >= 8 else uttid
    
    # 确保输出目录存在
    output_dir = os.path.join(output_root, subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名：uttid.seg索引_起始时间_结束时间.wav
    filename = f"{uttid}.seg{seg_idx:03d}_{start_ms}_{end_ms}.wav"
    output_path = os.path.join(output_dir, filename)
    
    return output_path

def merge_segments_by_second_dimension(segment_label, seg_timestamp_list, min_second_dim_length=2):
    """
    将segment_label第一个维度上相邻元素适当组合
    条件：合并后第二个维度长度之和 >= min_second_dim_length
    策略：
    1. 避免最后一个元素单独成组且未达阈值
    2. 一旦达到阈值，在满足需求1的前提下，就不要再继续组合
    segment_label: 三维数组 [[[a,b], [c,d]], [[e]], [[f,g]], ...]
    min_second_dim_length: 第二个维度长度阈值，默认为2
    返回: [(merged_segment_label, merged_timestamp, start_idx, end_idx), ...]
    """
    merged_segments = []
    i = 0
    
    while i < len(segment_label):
        # 从当前位置开始，尝试合并相邻的segments
        current_merged = []
        current_timestamp = []
        current_second_dim_total_length = 0
        start_idx = i
        
        # 计算剩余元素的总长度，用于判断是否需要继续合并
        remaining_length = sum(len(segment_label[j]) for j in range(i, len(segment_label)))
        
        # 合并segments直到第二个维度长度之和 >= min_second_dim_length
        while i < len(segment_label):
            # 将当前segment添加到合并列表中
            current_merged.append(segment_label[i])
            current_timestamp.append(seg_timestamp_list[i])
            
            # 计算当前segment的第二个维度长度（即len(segment_label[i])）
            current_seg_second_dim_length = len(segment_label[i])
            current_second_dim_total_length += current_seg_second_dim_length
            
            # 检查是否达到阈值
            if current_second_dim_total_length >= min_second_dim_length:
                # 达到阈值，检查是否需要继续合并以满足需求1
                remaining_after_current = sum(len(segment_label[j]) for j in range(i + 1, len(segment_label)))
                
                # 如果剩余元素长度 < 阈值，需要继续合并以避免最后一个元素单独成组
                if remaining_after_current > 0 and remaining_after_current < min_second_dim_length:
                    # 继续合并下一个元素
                    i += 1
                    continue
                else:
                    # 可以安全停止合并
                    break
            i += 1
        
        # 创建合并
        merged_segments.append((
            current_merged,
            current_timestamp,
            start_idx,
            i
        ))
        i += 1
    
    return merged_segments

def split_datalist_by_segments(datalist_dict, seg_timestamp_dict, output_audio_root, min_second_dim_length=2, audio_extension_ms=0):
    """
    按照合并后的segment分割datalist，生成多个segment的datalist条目
    min_second_dim_length: 第二个维度长度阈值，默认为2
    audio_extension_ms: 音频拓展长度（毫秒），默认为0
    """
    cut_datalist_entries = []
    
    # 确保输出根目录存在
    os.makedirs(output_audio_root, exist_ok=True)
    
    for uttid, utt_datalist_dict in datalist_dict.items():
        if uttid not in seg_timestamp_dict:
            continue
            
        seg_timestamp_list = seg_timestamp_dict[uttid]
        segment_label = utt_datalist_dict["segment_label"]
        original_sph = utt_datalist_dict["sph"]
        
        # 获取所有list类型的字段
        list_fields = {}
        for key, value in utt_datalist_dict.items():
            if isinstance(value, list) and key != "segment_label":
                list_fields[key] = value
        
        # 创建合并的segments
        merged_segments = merge_segments_by_second_dimension(segment_label, seg_timestamp_list, min_second_dim_length)
        
        # 获取音频文件时长（用于边界处理）
        audio_duration_ms = None
        if os.path.exists(original_sph):
            audio_duration_ms = get_audio_duration_ms(original_sph)
        
        # 为每个合并的segment创建一个新的datalist条目
        for seg_idx, (merged_segment_label, merged_timestamp, start_idx, end_idx) in enumerate(merged_segments):
            # 计算合并segment的总长度（所有内部数组长度之和）
            total_length = sum(sum(len(inner_list) for inner_list in seg) for seg in merged_segment_label)
            
            # 获取时间范围（应用拓展）
            start_ms, end_ms = get_segment_time_range(merged_timestamp, audio_extension_ms, audio_duration_ms)
            
            # 创建新的条目
            new_entry = {}
            
            # 复制非list字段
            for key, value in utt_datalist_dict.items():
                if not isinstance(value, list):
                    new_entry[key] = value
            
            # 更新key，添加segment后缀和时间戳信息
            new_entry["key"] = f"{uttid}.seg{seg_idx:03d}_{start_ms}_{end_ms}"
            
            # 生成切分后的音频文件路径（统一放入根目录）
            new_sph = generate_output_audio_path(uttid, seg_idx, start_ms, end_ms, output_audio_root)
            new_entry["sph"] = new_sph
            
            # 切分音频文件
            if os.path.exists(original_sph):
                success = cut_audio_file(original_sph, new_sph, start_ms, end_ms)
                if not success:
                    print(f"Failed to cut audio for {uttid} segment {seg_idx}")
            else:
                print(f"Warning: Original audio file not found: {original_sph}")
            
            # 分割list字段 - 根据合并的segments计算索引范围
            char_start_idx = sum(sum(len(inner_list) for inner_list in segment_label[j]) for j in range(start_idx))
            char_end_idx = char_start_idx + total_length
            
            for field_name, field_value in list_fields.items():
                if field_name == "phn_label":
                    # phn_label是二维list，需要特殊处理
                    new_entry[field_name] = field_value[char_start_idx:char_end_idx]
                else:
                    # 其他list字段直接按长度分割
                    new_entry[field_name] = field_value[char_start_idx:char_end_idx]
            
            # 添加合并的segment_label
            new_entry["segment_label"] = merged_segment_label
            
            # 添加合并的segment timestamp
            new_entry["segment_timestamp"] = merged_timestamp
            
            # 添加时间范围信息
            new_entry["start_time_ms"] = start_ms
            new_entry["end_time_ms"] = end_ms
            new_entry["original_start_time_ms"] = merged_timestamp[0][0][0]  # 原始起始时间
            new_entry["original_end_time_ms"] = merged_timestamp[-1][-1][1]  # 原始结束时间
            new_entry["extension_ms"] = audio_extension_ms
            
            # 添加合并信息
            new_entry["merged_segment_count"] = len(merged_segment_label)
            new_entry["total_length"] = total_length
            new_entry["original_segment_range"] = f"{start_idx}-{end_idx-1}"
            
            cut_datalist_entries.append(new_entry)
    
    return cut_datalist_entries

# 生成segment timestamp文件
with open(segment_timestamp, 'w') as f_seg_timestamp:
    seg_timestamp_dict = get_segment_timestamp(datalist_dict, c_timestamp_dict)
    for uttid, seg_timestamp_list in seg_timestamp_dict.items():
        f_seg_timestamp.write("{} {}\n".format(uttid, seg_timestamp_list))

# 生成分割后的datalist文件
with open(cut_datalist, 'w') as f_cut_datalist:
    cut_datalist_entries = split_datalist_by_segments(datalist_dict, seg_timestamp_dict, output_audio_root, min_second_dim_length, audio_extension_ms)
    for entry in cut_datalist_entries:
        f_cut_datalist.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Generated {len(cut_datalist_entries)} segment entries in {cut_datalist}")
print(f"Audio segments saved to: {output_audio_root}")
print(f"Using minimum second dimension length threshold: {min_second_dim_length}")
print(f"Using audio extension: {audio_extension_ms}ms")
print("Audio cutting completed. Please ensure ffmpeg and ffprobe are installed for audio processing.")




