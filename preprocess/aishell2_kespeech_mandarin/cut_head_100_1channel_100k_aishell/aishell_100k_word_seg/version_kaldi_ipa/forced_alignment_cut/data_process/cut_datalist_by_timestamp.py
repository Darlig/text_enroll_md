#!/usr/bin/env python3

import sys
import re
import json
import ast
import os
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from functools import partial

source_datalist = sys.argv[1]
character_timestamp = sys.argv[2]
segment_timestamp = sys.argv[3]
cut_datalist = sys.argv[4]
output_audio_root = sys.argv[5]  # 新增：分割后音频文件的根目录
min_second_dim_length = int(sys.argv[6]) if len(sys.argv) > 6 else 2  # 可配置的第二个维度长度阈值，默认为2
audio_extension_ms = int(sys.argv[7]) if len(sys.argv) > 7 else 0  # 音频拓展长度（毫秒），默认为0
max_workers = int(sys.argv[8]) if len(sys.argv) > 8 else min(8, mp.cpu_count())  # 并行处理的工作进程数，默认为CPU核心数
skip_duration_check = (len(sys.argv) > 9 and sys.argv[9].lower() == 'true') or audio_extension_ms == 0  # 跳过音频时长检查，默认当extension_ms=0时跳过

#1001501_26b0ce87 [[270, 450], [450, 490], [490, 590], [590, 650], [650, 730], [730, 770], [770, 830], [830, 890], [890, 930], [930, 990], [990, 1030], [1030, 1150], [1150, 1465]]
#{"key": "Y0000003589_8WOJb2iiULs_S00222", "sph": "/work104/weiyang/data/wenetspeech/dataset/drama_samp500h/audio_cut3/audio/train/youtube_wav/B00014/Y0000003589_8WOJb2iiULs/Y0000003589_8WOJb2iiULs_S00222.wav", "bpe_label": [815, 47, 484, 13, 14, 1566, 754, 22, 1732, 1708, 765], "phn_label": [[64, 93], [58, 143], [32, 88], [20, 21], [22, 19], [7, 184], [10, 113], [3, 61], [44, 91], [72, 52], [64, 93]], "segment_label": [[[64, 93], [58, 143], [32, 88]], [[20, 21], [22, 19]], [[7, 184], [10, 113]], [[3, 61], [44, 91]], [[72, 52], [64, 93]]], "kw_candidate": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b_kw_candidate": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

datalist_dict = {}
print(f"Reading source datalist from: {source_datalist}")
with open(source_datalist) as f_datalist:
    line_count = 0
    for line in f_datalist:
        line_count += 1
        if line_count % 10000 == 0:
            print(f"  Processed {line_count} datalist entries...")
        utt_datalist_dict = json.loads(line.strip())
        uttid = utt_datalist_dict["key"]
        datalist_dict[uttid] = utt_datalist_dict
print(f"Completed reading {len(datalist_dict)} datalist entries")

c_timestamp_dict = {}
print(f"Reading character timestamps from: {character_timestamp}")
with open(character_timestamp) as f_timestamp:
    line_count = 0
    for line in f_timestamp:
        line_count += 1
        if line_count % 10000 == 0:
            print(f"  Processed {line_count} timestamp entries...")
        uttid, content = re.split(r'\s+', line.strip(), maxsplit=1)
        c_timestamp = ast.literal_eval(content)
        c_timestamp_dict[uttid] = c_timestamp
print(f"Completed reading {len(c_timestamp_dict)} timestamp entries")

def process_single_utterance_timestamp(args):
    """
    处理单个utterance的timestamp生成，用于并行处理
    """
    uttid, utt_datalist_dict, c_timestamp_dict = args
    if uttid in c_timestamp_dict:
        c_timestamp = c_timestamp_dict[uttid]
        segment_label = utt_datalist_dict["segment_label"]
        seg_len_list = [len(seg) for seg in segment_label]
        seg_timestamp_list = []
        i = 0
        for sl in seg_len_list:
            seg_ts = c_timestamp[i:i+sl]
            seg_timestamp_list.append(seg_ts)
            i += sl
        return uttid, seg_timestamp_list
    return uttid, None

def process_utterance_batch_timestamp(args):
    """
    批量处理多个utterance的timestamp生成，减少进程间通信开销
    """
    batch_tasks, c_timestamp_dict = args
    batch_results = []
    
    for uttid, utt_datalist_dict in batch_tasks:
        if uttid in c_timestamp_dict:
            c_timestamp = c_timestamp_dict[uttid]
            segment_label = utt_datalist_dict["segment_label"]
            seg_len_list = [len(seg) for seg in segment_label]
            seg_timestamp_list = []
            i = 0
            for sl in seg_len_list:
                seg_ts = c_timestamp[i:i+sl]
                seg_timestamp_list.append(seg_ts)
                i += sl
            batch_results.append((uttid, seg_timestamp_list))
        else:
            batch_results.append((uttid, None))
    
    return batch_results

def get_segment_timestamp(datalist_dict, c_timestamp_dict, max_workers=None):
    """
    并行处理segment timestamp生成（智能优化版本）
    """
    if max_workers is None:
        max_workers = min(8, mp.cpu_count())
    
    datalist_items = list(datalist_dict.items())
    total_tasks = len(datalist_items)
    
    print(f"Generating segment timestamps for {total_tasks} utterances using {max_workers} workers...")
    
    seg_timestamp_dict = {}
    
    if total_tasks > 1 and max_workers > 1:
        # 根据任务数量选择处理策略
        if total_tasks > 5000:
            # 大量任务：使用批处理减少进程间通信开销
            batch_size = max(50, total_tasks // (max_workers * 4))
            print(f"  Using batch processing (batch size: {batch_size}) for large dataset...")
            
            # 创建批次
            batches = []
            for i in range(0, total_tasks, batch_size):
                batch_tasks = datalist_items[i:i + batch_size]
                batches.append((batch_tasks, c_timestamp_dict))
            
            print(f"  Created {len(batches)} batches for processing")
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 使用submit方式提交批次任务
                future_to_batch = {executor.submit(process_utterance_batch_timestamp, batch): i 
                                 for i, batch in enumerate(batches)}
                
                # 实时收集结果并显示进度
                processed_batches = 0
                processed_utterances = 0
                
                for future in future_to_batch:
                    try:
                        batch_results = future.result(timeout=120)  # 批处理超时时间更长
                        processed_batches += 1
                        
                        # 处理批次结果
                        for uttid, seg_timestamp_list in batch_results:
                            processed_utterances += 1
                            if seg_timestamp_list is not None:
                                seg_timestamp_dict[uttid] = seg_timestamp_list
                        
                        # 显示进度
                        if processed_batches % max(1, len(batches) // 20) == 0 or processed_batches == len(batches):
                            elapsed_time = time.time() - start_time
                            progress_percent = (processed_batches / len(batches)) * 100
                            avg_time_per_batch = elapsed_time / processed_batches
                            estimated_remaining = avg_time_per_batch * (len(batches) - processed_batches)
                            
                            print(f"  Progress: {processed_batches}/{len(batches)} batches ({progress_percent:.1f}%) - "
                                  f"Processed {processed_utterances} utterances - "
                                  f"Elapsed: {elapsed_time:.1f}s, ETA: {estimated_remaining:.1f}s, "
                                  f"Speed: {processed_utterances/elapsed_time:.1f} utterances/sec")
                            
                    except Exception as e:
                        batch_idx = future_to_batch[future]
                        print(f"  Warning: Batch {batch_idx} failed with error: {e}")
                        processed_batches += 1
        else:
            # 中等任务量：使用单任务处理方式
            print(f"  Using individual task processing for medium dataset...")
            tasks = [(uttid, utt_datalist_dict, c_timestamp_dict) 
                     for uttid, utt_datalist_dict in datalist_items]
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 使用submit方式提交任务，可以实时获取结果
                future_to_task = {executor.submit(process_single_utterance_timestamp, task): i 
                                 for i, task in enumerate(tasks)}
                
                # 实时收集结果并显示进度
                processed_count = 0
                for future in future_to_task:
                    try:
                        uttid, seg_timestamp_list = future.result(timeout=60)  # 60秒超时
                        processed_count += 1
                        
                        if seg_timestamp_list is not None:
                            seg_timestamp_dict[uttid] = seg_timestamp_list
                        
                        # 显示详细进度信息
                        if processed_count % 500 == 0 or processed_count % max(1, len(tasks) // 20) == 0:
                            elapsed_time = time.time() - start_time
                            progress_percent = (processed_count / len(tasks)) * 100
                            avg_time_per_task = elapsed_time / processed_count
                            estimated_remaining = avg_time_per_task * (len(tasks) - processed_count)
                            
                            print(f"  Progress: {processed_count}/{len(tasks)} ({progress_percent:.1f}%) - "
                                  f"Elapsed: {elapsed_time:.1f}s, ETA: {estimated_remaining:.1f}s, "
                                  f"Speed: {processed_count/elapsed_time:.1f} tasks/sec")
                            
                    except Exception as e:
                        task_idx = future_to_task[future]
                        print(f"  Warning: Task {task_idx} failed with error: {e}")
                        processed_count += 1
    else:
        # 串行处理（用于调试或小数据集）
        print(f"  Using serial processing for small dataset...")
        processed_count = 0
        start_time = time.time()
        for uttid, utt_datalist_dict in datalist_items:
            processed_count += 1
            if processed_count % 1000 == 0:
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / processed_count
                eta = avg_time * (total_tasks - processed_count)
                print(f"  Processed {processed_count}/{total_tasks} timestamp generations... "
                      f"Elapsed: {elapsed_time:.1f}s, ETA: {eta:.1f}s")
            
            if uttid in c_timestamp_dict:
                c_timestamp = c_timestamp_dict[uttid]
                segment_label = utt_datalist_dict["segment_label"]
                seg_len_list = [len(seg) for seg in segment_label]
                seg_timestamp_list = []
                i = 0
                for sl in seg_len_list:
                    seg_ts = c_timestamp[i:i+sl]
                    seg_timestamp_list.append(seg_ts)
                    i += sl
                seg_timestamp_dict[uttid] = seg_timestamp_list
    
    print(f"Completed segment timestamp generation for {len(seg_timestamp_dict)} utterances")
    return seg_timestamp_dict

def get_segment_time_range(seg_timestamp, extension_ms=0, audio_duration_ms=None):
    """
    从segment timestamp中获取起止时间（毫秒），并应用拓展
    seg_timestamp: [[start1, end1], [start2, end2], ...] 或嵌套结构
    extension_ms: 拓展长度（毫秒）
    audio_duration_ms: 音频总长度（毫秒），用于边界处理
    返回: (start_time_ms, end_time_ms)
    """
    if not seg_timestamp:
        return 0, 0
    
    try:
        # 递归函数来提取所有时间值
        def extract_all_times(data):
            """递归提取所有时间戳值"""
            times = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, list):
                        if len(item) >= 2 and all(isinstance(x, (int, float)) for x in item[:2]):
                            # 这是一个时间对 [start, end]
                            times.extend(item[:2])
                        else:
                            # 继续递归
                            times.extend(extract_all_times(item))
                    elif isinstance(item, (int, float)):
                        times.append(item)
            return times
        
        # 提取所有时间值
        all_times = extract_all_times(seg_timestamp)
        
        if not all_times:
            print(f"Warning: No valid time values found in seg_timestamp: {seg_timestamp}")
            return 0, 0
        
        # 获取最小和最大时间
        original_start_time = min(all_times)
        original_end_time = max(all_times)
        
        # 应用拓展
        start_time = max(0, original_start_time - extension_ms)  # 不能小于0
        end_time = original_end_time + extension_ms
        
        # 如果提供了音频总长度，确保不超过音频边界
        if audio_duration_ms is not None:
            end_time = min(end_time, audio_duration_ms)
        
        return start_time, end_time
        
    except Exception as e:
        print(f"Error processing seg_timestamp {seg_timestamp}: {e}")
        # 返回默认值或尝试简单的fallback
        try:
            if isinstance(seg_timestamp, list) and len(seg_timestamp) > 0:
                # 尝试最简单的访问模式
                first_item = seg_timestamp[0]
                last_item = seg_timestamp[-1]
                
                # 尝试提取第一个和最后一个数值
                if isinstance(first_item, (int, float)):
                    start_val = first_item
                elif isinstance(first_item, list) and len(first_item) > 0:
                    start_val = first_item[0] if isinstance(first_item[0], (int, float)) else 0
                else:
                    start_val = 0
                    
                if isinstance(last_item, (int, float)):
                    end_val = last_item
                elif isinstance(last_item, list) and len(last_item) > 0:
                    end_val = last_item[-1] if isinstance(last_item[-1], (int, float)) else start_val + 1000
                else:
                    end_val = start_val + 1000
                
                return max(0, start_val - extension_ms), end_val + extension_ms
        except:
            pass
        
        # 最后的fallback
        print(f"Using fallback values for problematic timestamp: {seg_timestamp}")
        return 0, 1000

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

def get_audio_duration_batch(audio_paths, max_workers=None):
    """
    并行获取多个音频文件的时长
    """
    if max_workers is None:
        max_workers = min(8, mp.cpu_count())
    
    # 去重
    unique_paths = list(set(audio_paths))
    print(f"Getting durations for {len(unique_paths)} unique audio files using {max_workers} workers...")
    
    if len(unique_paths) <= 1 or max_workers <= 1:
        # 串行处理
        duration_dict = {}
        processed_count = 0
        for path in unique_paths:
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"  Processed duration for {processed_count}/{len(unique_paths)} audio files...")
            if os.path.exists(path):
                duration_dict[path] = get_audio_duration_ms(path)
        return duration_dict
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 只处理存在的文件
        existing_paths = [path for path in unique_paths if os.path.exists(path)]
        print(f"  Found {len(existing_paths)} existing audio files out of {len(unique_paths)} unique paths")
        
        # 提交任务
        future_to_path = {executor.submit(get_audio_duration_ms, path): path 
                         for path in existing_paths}
        
        # 收集结果
        duration_dict = {}
        completed_count = 0
        for future in future_to_path:
            path = future_to_path[future]
            completed_count += 1
            if completed_count % 500 == 0:
                print(f"  Completed duration for {completed_count}/{len(existing_paths)} audio files...")
            try:
                duration = future.result()
                duration_dict[path] = duration
            except Exception as e:
                print(f"Error getting duration for {path}: {e}")
                duration_dict[path] = None
    
    print(f"Completed getting durations for {len(duration_dict)} audio files")
    return duration_dict

def cut_audio_file(input_audio_path, output_audio_path, start_ms, end_ms):
    """
    使用ffmpeg切分音频文件
    start_ms, end_ms: 起止时间（毫秒）
    """
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

def process_single_audio_segment(args):
    """
    处理单个音频segment的切分
    用于并行处理
    """
    (input_audio_path, output_audio_path, start_ms, end_ms, uttid, seg_idx) = args
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    
    # 切分音频
    success = cut_audio_file(input_audio_path, output_audio_path, start_ms, end_ms)
    
    return {
        'uttid': uttid,
        'seg_idx': seg_idx,
        'success': success,
        'output_path': output_audio_path
    }

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

def process_utterance_chunk(args):
    """
    处理一批utterance的数据准备，用于并行处理
    """
    chunk_data, seg_timestamp_dict, audio_duration_cache, min_second_dim_length, audio_extension_ms, output_audio_root = args
    
    chunk_entries = []
    chunk_audio_tasks = []
    processed_count = 0
    
    for uttid, utt_datalist_dict in chunk_data.items():
        processed_count += 1
        # 为避免过多输出，只在每个chunk的第一个元素时输出进度
        if processed_count == 1:
            print(f"  Processing chunk with {len(chunk_data)} utterances...")
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
        
        # 从缓存中获取音频文件时长
        audio_duration_ms = audio_duration_cache.get(original_sph)
        
        # 为每个合并的segment准备数据
        for seg_idx, (merged_segment_label, merged_timestamp, start_idx, end_idx) in enumerate(merged_segments):
            try:
                # 数据验证
                if not merged_segment_label or not merged_timestamp:
                    print(f"Warning: Empty segment data for {uttid} segment {seg_idx}, skipping...")
                    continue
                
                # 计算合并segment的总长度（所有内部数组长度之和）
                total_length = sum(sum(len(inner_list) for inner_list in seg) for seg in merged_segment_label)
                
                # 获取时间范围（应用拓展）
                start_ms, end_ms = get_segment_time_range(merged_timestamp, audio_extension_ms, audio_duration_ms)
                
                # 验证时间范围的合理性
                if start_ms < 0 or end_ms <= start_ms:
                    print(f"Warning: Invalid time range for {uttid} segment {seg_idx}: {start_ms}-{end_ms}ms, using fallback...")
                    # 使用fallback值
                    start_ms, end_ms = 0, 1000
                    
            except Exception as e:
                print(f"Error processing {uttid} segment {seg_idx}: {e}")
                print(f"  merged_segment_label type: {type(merged_segment_label)}")
                print(f"  merged_timestamp type: {type(merged_timestamp)}")
                print(f"  merged_timestamp content: {merged_timestamp}")
                # 使用fallback值继续处理
                start_ms, end_ms = 0, 1000
                total_length = 1
            
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
            
            # 准备音频切分任务
            if os.path.exists(original_sph):
                chunk_audio_tasks.append((original_sph, new_sph, start_ms, end_ms, uttid, seg_idx))
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
            
            # 安全地提取原始时间信息
            try:
                # 使用与get_segment_time_range相同的逻辑提取时间
                def extract_all_times(data):
                    times = []
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, list):
                                if len(item) >= 2 and all(isinstance(x, (int, float)) for x in item[:2]):
                                    times.extend(item[:2])
                                else:
                                    times.extend(extract_all_times(item))
                            elif isinstance(item, (int, float)):
                                times.append(item)
                    return times
                
                all_times = extract_all_times(merged_timestamp)
                if all_times:
                    new_entry["original_start_time_ms"] = min(all_times)
                    new_entry["original_end_time_ms"] = max(all_times)
                else:
                    new_entry["original_start_time_ms"] = start_ms
                    new_entry["original_end_time_ms"] = end_ms
                    print(f"Warning: Could not extract original times for {uttid}, using calculated times")
            except Exception as e:
                print(f"Warning: Error extracting original times for {uttid}: {e}")
                new_entry["original_start_time_ms"] = start_ms
                new_entry["original_end_time_ms"] = end_ms
            
            new_entry["extension_ms"] = audio_extension_ms
            
            # 添加合并信息
            new_entry["merged_segment_count"] = len(merged_segment_label)
            new_entry["total_length"] = total_length
            new_entry["original_segment_range"] = f"{start_idx}-{end_idx-1}"
            
            chunk_entries.append(new_entry)
    
    return chunk_entries, chunk_audio_tasks

def split_datalist_by_segments(datalist_dict, seg_timestamp_dict, output_audio_root, min_second_dim_length=2, audio_extension_ms=0, max_workers=4, skip_duration_check=True):
    """
    按照合并后的segment分割datalist，生成多个segment的datalist条目
    min_second_dim_length: 第二个维度长度阈值，默认为2
    audio_extension_ms: 音频拓展长度（毫秒），默认为0
    max_workers: 并行处理的工作进程数
    skip_duration_check: 是否跳过音频时长检查，默认为True
    """
    cut_datalist_entries = []
    audio_tasks = []  # 存储音频切分任务
    
    # 确保输出根目录存在
    os.makedirs(output_audio_root, exist_ok=True)
    
    print(f"Processing {len(datalist_dict)} utterances...")
    
    # 第一步：根据设置决定是否获取音频时长
    if skip_duration_check:
        print("Skipping audio duration check (extension_ms=0 or explicitly disabled)")
        audio_duration_cache = {}
    else:
        # 收集所有需要获取时长的音频文件
        all_audio_paths = []
        for uttid, utt_datalist_dict in datalist_dict.items():
            if uttid in seg_timestamp_dict:
                original_sph = utt_datalist_dict["sph"]
                if os.path.exists(original_sph):
                    all_audio_paths.append(original_sph)
        
        # 并行获取所有音频文件的时长
        print(f"Getting duration for {len(set(all_audio_paths))} unique audio files...")
        audio_duration_cache = get_audio_duration_batch(all_audio_paths, max_workers)
    
    # 第三步：将数据分块并并行处理数据准备
    chunk_size = max(1, len(datalist_dict) // max_workers) if len(datalist_dict) > max_workers else len(datalist_dict)
    
    # 将数据分成chunks
    datalist_items = list(datalist_dict.items())
    chunks = []
    for i in range(0, len(datalist_items), chunk_size):
        chunk_dict = dict(datalist_items[i:i + chunk_size])
        chunks.append((chunk_dict, seg_timestamp_dict, audio_duration_cache, 
                      min_second_dim_length, audio_extension_ms, output_audio_root))
    
    print(f"Processing data in {len(chunks)} chunks with {max_workers} workers...")
    
    if len(chunks) > 1 and max_workers > 1:
        # 并行处理数据准备
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(process_utterance_chunk, chunks))
        
        # 收集所有结果
        processed_chunks = 0
        for chunk_entries, chunk_audio_tasks in chunk_results:
            processed_chunks += 1
            print(f"  Completed data processing for chunk {processed_chunks}/{len(chunks)}")
            cut_datalist_entries.extend(chunk_entries)
            audio_tasks.extend(chunk_audio_tasks)
    else:
        # 串行处理（用于小数据集或调试）
        processed_chunks = 0
        for chunk in chunks:
            processed_chunks += 1
            print(f"  Processing chunk {processed_chunks}/{len(chunks)}...")
            chunk_entries, chunk_audio_tasks = process_utterance_chunk(chunk)
            cut_datalist_entries.extend(chunk_entries)
            audio_tasks.extend(chunk_audio_tasks)
    
    print(f"Completed data processing. Generated {len(cut_datalist_entries)} entries and {len(audio_tasks)} audio tasks")
    
    # 第二遍：并行处理音频切分
    if audio_tasks:
        print(f"Starting audio cutting for {len(audio_tasks)} segments with {max_workers} workers...")
        start_time = time.time()
        
        # 为了更好的进度输出，我们使用submit方式而不map，这样可以实时输出进度
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_single_audio_segment, task) for task in audio_tasks]
            
            # 收集结果并显示进度
            results = []
            completed_count = 0
            for future in futures:
                result = future.result()
                results.append(result)
                completed_count += 1
                
                # 每处理100个或者每10%输出一次进度
                if completed_count % 100 == 0 or completed_count % max(1, len(audio_tasks) // 10) == 0:
                    progress_percent = (completed_count / len(audio_tasks)) * 100
                    elapsed_time = time.time() - start_time
                    avg_time_per_task = elapsed_time / completed_count
                    estimated_remaining = avg_time_per_task * (len(audio_tasks) - completed_count)
                    print(f"  Audio cutting progress: {completed_count}/{len(audio_tasks)} ({progress_percent:.1f}%) - "
                          f"Elapsed: {elapsed_time:.1f}s, ETA: {estimated_remaining:.1f}s")
        
        # 统计结果
        successful_cuts = sum(1 for r in results if r['success'])
        failed_cuts = len(results) - successful_cuts
        
        end_time = time.time()
        print(f"Audio processing completed in {end_time - start_time:.2f} seconds")
        print(f"Successfully cut {successful_cuts} segments, failed {failed_cuts} segments")
        
        if failed_cuts > 0:
            print("Failed segments:")
            for r in results:
                if not r['success']:
                    print(f"  {r['uttid']} segment {r['seg_idx']}")
    
    return cut_datalist_entries

# 生成segment timestamp文件
print(f"\n=== Step 1: Generating segment timestamps ===")
with open(segment_timestamp, 'w') as f_seg_timestamp:
    seg_timestamp_dict = get_segment_timestamp(datalist_dict, c_timestamp_dict, max_workers)
    print(f"Writing {len(seg_timestamp_dict)} segment timestamps to file...")
    written_count = 0
    for uttid, seg_timestamp_list in seg_timestamp_dict.items():
        f_seg_timestamp.write("{} {}\n".format(uttid, seg_timestamp_list))
        written_count += 1
        if written_count % 10000 == 0:
            print(f"  Written {written_count}/{len(seg_timestamp_dict)} timestamp entries...")
print(f"Completed writing segment timestamps to: {segment_timestamp}")

# 生成分割后的datalist文件
print(f"\n=== Step 2: Processing datalist and cutting audio ===")
start_time = time.time()

with open(cut_datalist, 'w') as f_cut_datalist:
    cut_datalist_entries = split_datalist_by_segments(datalist_dict, seg_timestamp_dict, output_audio_root, min_second_dim_length, audio_extension_ms, max_workers, skip_duration_check)
    print(f"Writing {len(cut_datalist_entries)} datalist entries to file...")
    written_count = 0
    for entry in cut_datalist_entries:
        f_cut_datalist.write(json.dumps(entry, ensure_ascii=False) + '\n')
        written_count += 1
        if written_count % 5000 == 0:
            print(f"  Written {written_count}/{len(cut_datalist_entries)} datalist entries...")
    print(f"Completed writing datalist to: {cut_datalist}")

end_time = time.time()
print(f"\n=== Processing Summary ===")
print(f"Total processing time: {end_time - start_time:.2f} seconds")
print(f"Generated {len(cut_datalist_entries)} segment entries in {cut_datalist}")
print(f"Audio segments saved to: {output_audio_root}")
print(f"Using minimum second dimension length threshold: {min_second_dim_length}")
print(f"Using audio extension: {audio_extension_ms}ms")
print(f"Audio duration check: {'Disabled' if skip_duration_check else 'Enabled'}")
print(f"Using {max_workers} parallel workers")
print("Audio cutting completed. Please ensure ffmpeg and ffprobe are installed for audio processing.")




