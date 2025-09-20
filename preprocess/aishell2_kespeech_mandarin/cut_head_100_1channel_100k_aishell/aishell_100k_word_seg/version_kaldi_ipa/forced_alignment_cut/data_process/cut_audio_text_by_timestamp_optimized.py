import json
import os
from typing import List, Tuple, Dict, Any
import librosa
import soundfile as sf
import numpy as np
import sys
import re
import ast
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial
import time

class AudioSegmentationToolOptimized:
    def __init__(self, datalist_path: str, timestamp_path: str, text_path: str = None, output_dir: str = "output_segments"):
        """
        优化版音频切割工具
        
        Args:
            datalist_path: 包含分词信息的datalist.txt文件路径
            timestamp_path: 包含时间戳信息的timestamp.txt文件路径
            text_path: 包含音频文本的文件路径（可选）
            output_dir: 输出目录
        """
        self.datalist_path = datalist_path
        self.timestamp_path = timestamp_path
        self.text_path = text_path
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        print("加载数据中...")
        start_time = time.time()
        self.datalist = self._load_datalist()
        self.timestamps = self._load_timestamps()
        self.text_data = self._load_text_data() if text_path else {}
        print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
        
    def _load_datalist(self) -> Dict[str, Any]:
        """加载分词信息"""
        datalist = {}
        with open(self.datalist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        datalist[json_data['key']] = json_data
                    except json.JSONDecodeError:
                        continue
        return datalist
    
    def _load_timestamps(self) -> Dict[str, List[List[int]]]:
        """加载时间戳信息"""
        timestamps = {}
        with open(self.timestamp_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        parts = re.split(r'\s+', line, maxsplit=1)
                        key = parts[0]
                        timestamp_data = parts[1]
                        timestamps[key] = ast.literal_eval(timestamp_data)
                    except (ValueError, SyntaxError):
                        continue
        return timestamps
    
    def _load_text_data(self) -> Dict[str, str]:
        """加载音频文本数据"""
        text_data = {}
        with open(self.text_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        uttid, text_content = parts
                        text_data[uttid] = text_content
        return text_data
    
    def _calculate_word_character_count(self, segment_label: List[List[List[int]]]) -> List[int]:
        """计算每个词的字数"""
        return [len(word) for word in segment_label]
    
    def _group_words_by_threshold(self, word_char_counts: List[int], threshold: int) -> List[Tuple[int, int]]:
        """根据阈值将词分组，确保每个片段都达到字数阈值"""
        groups = []
        current_char_count = 0
        start_word_idx = 0
        
        i = 0
        while i < len(word_char_counts):
            char_count = word_char_counts[i]
            current_char_count += char_count
            
            if current_char_count >= threshold or i == len(word_char_counts) - 1:
                groups.append((start_word_idx, i))
                start_word_idx = i + 1
                current_char_count = 0
            
            i += 1
        
        # 如果最后一组字数不足且有多个组，将最后一组合并到倒数第二组
        if len(groups) >= 2:
            last_group_start, last_group_end = groups[-1]
            last_group_char_count = sum(word_char_counts[last_group_start:last_group_end + 1])
            
            if last_group_char_count < threshold:
                groups.pop()
                second_last_start, _ = groups[-1]
                groups[-1] = (second_last_start, last_group_end)
            
        return groups
    
    def _extract_text_segment(self, full_text: str, segment_label: List[List[List[int]]], 
                            start_word: int, end_word: int) -> str:
        """从完整文本中提取对应片段的文本"""
        if not full_text:
            return ""
        
        total_chars = sum(len(segment_label[i]) for i in range(start_word, end_word + 1))
        start_char_pos = sum(len(segment_label[i]) for i in range(start_word))
        end_char_pos = start_char_pos + total_chars
        
        if start_char_pos < len(full_text) and end_char_pos <= len(full_text):
            return full_text[start_char_pos:end_char_pos]
        elif start_char_pos < len(full_text):
            return full_text[start_char_pos:]
        else:
            return ""
    
    def _calculate_segment_timestamps(self, segment_label: List[List[List[int]]], 
                                    timestamps: List[List[int]], 
                                    word_groups: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """计算每个切割片段的时间戳"""
        segment_timestamps = []
        char_idx = 0
        
        for start_word, end_word in word_groups:
            start_char_idx = char_idx
            char_count = sum(len(segment_label[word_idx]) for word_idx in range(start_word, end_word + 1))
            end_char_idx = start_char_idx + char_count - 1
            
            if start_char_idx < len(timestamps) and end_char_idx < len(timestamps):
                start_time = timestamps[start_char_idx][0]
                end_time = timestamps[end_char_idx][1]
                segment_timestamps.append((start_time, end_time))
            
            char_idx += char_count
            
        return segment_timestamps


def process_single_audio(args):
    """处理单个音频文件的函数，用于多进程"""
    audio_key, datalist_item, timestamps, text_data, threshold, extend_before_ms, extend_after_ms, output_dir = args
    
    try:
        data = datalist_item
        timestamp_list = timestamps
        segment_label = data['segment_label']
        audio_path = data['sph']
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            return audio_key, []
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=None)
        
        # 计算分组
        tool = AudioSegmentationToolOptimized("", "", "", "")
        word_char_counts = tool._calculate_word_character_count(segment_label)
        word_groups = tool._group_words_by_threshold(word_char_counts, threshold)
        segment_timestamps = tool._calculate_segment_timestamps(segment_label, timestamp_list, word_groups)
        
        # 获取完整文本
        full_text = text_data.get(audio_key, "")
        
        # 切割音频并保存
        cut_info = []
        audio_duration_ms = len(audio) / sr * 1000
        
        for i, ((start_word, end_word), (start_time, end_time)) in enumerate(zip(word_groups, segment_timestamps)):
            # 应用延伸时间
            extended_start_time = max(0, start_time - extend_before_ms)
            extended_end_time = min(audio_duration_ms, end_time + extend_after_ms)
            
            # 转换时间戳为秒
            start_sec = start_time / 1000.0
            end_sec = end_time / 1000.0
            extended_start_sec = extended_start_time / 1000.0
            extended_end_sec = extended_end_time / 1000.0
            
            # 转换为样本索引
            start_sample = int(extended_start_sec * sr)
            end_sample = int(extended_end_sec * sr)
            
            if start_sample < len(audio) and end_sample <= len(audio):
                audio_segment = audio[start_sample:end_sample]
                
                # 生成输出文件名
                output_filename = f"{audio_key}_segment_{i:03d}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # 保存音频片段
                sf.write(output_path, audio_segment, sr)
                
                # 收集信息
                char_count = sum(len(segment_label[word_idx]) for word_idx in range(start_word, end_word + 1))
                text_segment = tool._extract_text_segment(full_text, segment_label, start_word, end_word)
                
                segment_info = {
                    'segment_id': i,
                    'filename': output_filename,
                    'file_path': output_path,
                    'word_range': (start_word, end_word),
                    'total_chars': char_count,
                    'text_segment': text_segment,
                    'original_time_range_ms': (start_time, end_time),
                    'extended_time_range_ms': (extended_start_time, extended_end_time),
                    'original_duration_sec': end_sec - start_sec,
                    'extended_duration_sec': extended_end_sec - extended_start_sec,
                    'extend_before_ms': extend_before_ms,
                    'extend_after_ms': extend_after_ms
                }
                cut_info.append(segment_info)
        
        return audio_key, cut_info
        
    except Exception as e:
        print(f"处理音频 {audio_key} 失败: {e}")
        return audio_key, []


class AudioSegmentationToolOptimized:
    def __init__(self, datalist_path: str, timestamp_path: str, text_path: str = None, output_dir: str = "output_segments"):
        """优化版音频切割工具"""
        self.datalist_path = datalist_path
        self.timestamp_path = timestamp_path
        self.text_path = text_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("加载数据中...")
        start_time = time.time()
        self.datalist = self._load_datalist()
        self.timestamps = self._load_timestamps()
        self.text_data = self._load_text_data() if text_path else {}
        print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
        
    def _load_datalist(self) -> Dict[str, Any]:
        """加载分词信息"""
        datalist = {}
        with open(self.datalist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        datalist[json_data['key']] = json_data
                    except json.JSONDecodeError:
                        continue
        return datalist
    
    def _load_timestamps(self) -> Dict[str, List[List[int]]]:
        """加载时间戳信息"""
        timestamps = {}
        with open(self.timestamp_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        parts = re.split(r'\s+', line, maxsplit=1)
                        key = parts[0]
                        timestamp_data = parts[1]
                        timestamps[key] = ast.literal_eval(timestamp_data)
                    except (ValueError, SyntaxError):
                        continue
        return timestamps
    
    def _load_text_data(self) -> Dict[str, str]:
        """加载音频文本数据"""
        text_data = {}
        if not self.text_path:
            return text_data
        with open(self.text_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        uttid, text_content = parts
                        text_data[uttid] = text_content
        return text_data
    
    def cut_all_audio_parallel(self, threshold: int = 6, extend_before_ms: int = 0, 
                             extend_after_ms: int = 0, max_workers: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        并行处理所有音频
        
        Args:
            threshold: 每个片段的字数阈值
            extend_before_ms: 在片段开始前延伸的毫秒数
            extend_after_ms: 在片段结束后延伸的毫秒数
            max_workers: 最大工作进程数，默认为CPU核心数
            
        Returns:
            所有音频的切割信息
        """
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        print(f"使用 {max_workers} 个进程并行处理...")
        
        # 准备参数
        process_args = []
        for audio_key in self.datalist.keys():
            if audio_key in self.timestamps:
                args = (
                    audio_key,
                    self.datalist[audio_key],
                    self.timestamps[audio_key],
                    self.text_data,
                    threshold,
                    extend_before_ms,
                    extend_after_ms,
                    self.output_dir
                )
                process_args.append(args)
        
        # 使用进程池处理
        all_cut_info = {}
        total_files = len(process_args)
        processed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_single_audio, args) for args in process_args]
            
            # 收集结果
            for future in futures:
                try:
                    audio_key, cut_info = future.result()
                    all_cut_info[audio_key] = cut_info
                    processed += 1
                    
                    if processed % 10 == 0:
                        print(f"已处理: {processed}/{total_files} ({processed/total_files*100:.1f}%)")
                        
                except Exception as e:
                    print(f"处理失败: {e}")
                    processed += 1
        
        return all_cut_info
    
    def save_cut_info_streaming(self, cut_info: Dict[str, List[Dict[str, Any]]], 
                              info_filename: str = "cut_info.json"):
        """流式保存切割信息，避免内存峰值"""
        info_path = os.path.join(self.output_dir, info_filename)
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write('{\n')
            first = True
            for audio_key, segments in cut_info.items():
                if not first:
                    f.write(',\n')
                f.write(f'  "{audio_key}": ')
                json.dump(segments, f, ensure_ascii=False, indent=2)
                first = False
            f.write('\n}')
        print(f"切割信息已保存到: {info_path}")
    
    def save_text_segments(self, cut_info: Dict[str, List[Dict[str, Any]]], 
                         text_filename: str = "text_segments.txt"):
        """保存文本片段"""
        text_path = os.path.join(self.output_dir, text_filename)
        with open(text_path, 'w', encoding='utf-8') as f:
            for audio_key, segments in cut_info.items():
                for segment in segments:
                    segment_filename = segment['filename'].replace('.wav', '')
                    text_segment = segment.get('text_segment', '')
                    f.write(f"{segment_filename} {text_segment}\n")
        print(f"文本片段已保存到: {text_path}")


def main():
    """优化版主函数"""
    datalist_path = sys.argv[1]
    timestamp_path = sys.argv[2]
    text_path = sys.argv[3]
    output_dir = sys.argv[4]
    threshold = int(sys.argv[5])
    extend_before_ms = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    extend_after_ms = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    max_workers = int(sys.argv[8]) if len(sys.argv) > 8 else None
    
    print(f"配置参数:")
    print(f"  字数阈值: {threshold}")
    print(f"  前延伸: {extend_before_ms}ms")
    print(f"  后延伸: {extend_after_ms}ms")
    print(f"  最大进程数: {max_workers or multiprocessing.cpu_count()}")
    
    # 创建工具
    tool = AudioSegmentationToolOptimized(datalist_path, timestamp_path, text_path, output_dir)
    
    # 并行处理
    start_time = time.time()
    all_cut_info = tool.cut_all_audio_parallel(threshold, extend_before_ms, extend_after_ms, max_workers)
    processing_time = time.time() - start_time
    
    # 保存结果
    print("保存结果中...")
    tool.save_cut_info_streaming(all_cut_info)
    if text_path and tool.text_data:
        tool.save_text_segments(all_cut_info)
    
    # 统计信息
    total_segments = sum(len(info) for info in all_cut_info.values())
    successful_audio = sum(1 for info in all_cut_info.values() if len(info) > 0)
    segments_with_text = sum(1 for segments in all_cut_info.values() 
                           for segment in segments if segment.get('text_segment', ''))
    
    print(f"\n处理完成!")
    print(f"总耗时: {processing_time:.2f}秒")
    print(f"成功处理音频数: {successful_audio}")
    print(f"总切割片段数: {total_segments}")
    print(f"包含文本的片段数: {segments_with_text}")
    print(f"平均处理速度: {successful_audio/processing_time:.2f} 文件/秒")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
