import json
import os
from typing import List, Tuple, Dict, Any
import librosa
import soundfile as sf
import numpy as np
import sys
import re
import ast

class AudioSegmentationTool:
    def __init__(self, datalist_path: str, timestamp_path: str, text_path: str = None, output_dir: str = "output_segments"):
        """
        初始化音频切割工具
        
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
        self.datalist = self._load_datalist()
        self.timestamps = self._load_timestamps()
        self.text_data = self._load_text_data() if text_path else {}
        
    def _load_datalist(self) -> Dict[str, Any]:
        """加载分词信息"""
        datalist = {}
        with open(self.datalist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    
                    # 提取JSON数据
                    #parts = line.split(' ', 1)
                    #if len(parts) == 2:
                    #    line_num = parts[0]
                    #    print(parts[1])
                    #    #exit()
                        json_data = json.loads(line)
                        datalist[json_data['key']] = json_data
        return datalist
    
    def _load_timestamps(self) -> Dict[str, List[List[int]]]:
        """加载时间戳信息"""
        timestamps = {}
        with open(self.timestamp_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 提取key和时间戳数据
                    parts = re.split(r'\s+', line, maxsplit=1)
                    key = parts[0]
                    timestamp_data = parts[1]
                    timestamps[key] = ast.literal_eval(timestamp_data)
        return timestamps
    
    def _load_text_data(self) -> Dict[str, str]:
        """
        加载音频文本数据
        
        文件格式：每行为 "uttid 示例文本内容"
        
        Returns:
            字典，key为uttid，value为对应的文本内容
        """
        text_data = {}
        with open(self.text_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 分割uttid和文本内容，只分割第一个空格
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        uttid, text_content = parts
                        text_data[uttid] = text_content
        return text_data
    
    def _calculate_word_character_count(self, segment_label: List[List[List[int]]]) -> List[int]:
        """
        计算每个词的字数
        
        Args:
            segment_label: 分词信息，三级数组 [词][字][声母韵母]
            
        Returns:
            每个词的字数列表
        """
        return [len(word) for word in segment_label]
    
    def _group_words_by_threshold(self, word_char_counts: List[int], threshold: int) -> List[Tuple[int, int]]:
        """
        根据阈值将词分组，确保每个片段都达到字数阈值
        
        Args:
            word_char_counts: 每个词的字数列表
            threshold: 每个片段的字数阈值
            
        Returns:
            分组结果，每个元组包含(起始词索引, 结束词索引)
        """
        groups = []
        current_char_count = 0
        start_word_idx = 0
        
        i = 0
        while i < len(word_char_counts):
            char_count = word_char_counts[i]
            current_char_count += char_count
            
            # 如果达到或超过阈值，或者是最后一个词
            if current_char_count >= threshold or i == len(word_char_counts) - 1:
                groups.append((start_word_idx, i))
                # 开始新组
                start_word_idx = i + 1
                current_char_count = 0
            
            i += 1
        
        # 如果最后一组字数不足且有多个组，将最后一组合并到倒数第二组
        if len(groups) >= 2:
            last_group_start, last_group_end = groups[-1]
            last_group_char_count = sum(word_char_counts[last_group_start:last_group_end + 1])
            
            if last_group_char_count < threshold:
                # 移除最后一组，扩展倒数第二组
                groups.pop()
                second_last_start, _ = groups[-1]
                groups[-1] = (second_last_start, last_group_end)
            
        return groups
    
    def _extract_text_segment(self, full_text: str, segment_label: List[List[List[int]]], 
                            start_word: int, end_word: int) -> str:
        """
        从完整文本中提取对应片段的文本
        
        Args:
            full_text: 完整的文本内容
            segment_label: 分词信息
            start_word: 起始词索引
            end_word: 结束词索引
            
        Returns:
            提取的文本片段
        """
        if not full_text:
            return ""
        
        # 计算片段包含的总字数
        total_chars = sum(len(segment_label[i]) for i in range(start_word, end_word + 1))
        
        # 计算起始字符位置
        start_char_pos = sum(len(segment_label[i]) for i in range(start_word))
        
        # 提取对应的文本片段
        end_char_pos = start_char_pos + total_chars
        
        # 确保不超出文本边界
        if start_char_pos < len(full_text) and end_char_pos <= len(full_text):
            return full_text[start_char_pos:end_char_pos]
        elif start_char_pos < len(full_text):
            return full_text[start_char_pos:]
        else:
            return ""
    
    def _calculate_segment_timestamps(self, segment_label: List[List[List[int]]], 
                                    timestamps: List[List[int]], 
                                    word_groups: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        计算每个切割片段的时间戳
        
        Args:
            segment_label: 分词信息
            timestamps: 字级别时间戳
            word_groups: 词分组信息
            
        Returns:
            每个片段的时间戳 (开始时间, 结束时间)
        """
        segment_timestamps = []
        char_idx = 0  # 当前字的索引
        
        for start_word, end_word in word_groups:
            # 计算这个片段包含的字的索引范围
            start_char_idx = char_idx
            char_count = 0
            
            # 计算从start_word到end_word的总字数
            for word_idx in range(start_word, end_word + 1):
                char_count += len(segment_label[word_idx])
            
            end_char_idx = start_char_idx + char_count - 1
            
            # 获取时间戳
            if start_char_idx < len(timestamps) and end_char_idx < len(timestamps):
                # print(timestamps[start_char_idx])
                # print(timestamps[end_char_idx])
                start_time = timestamps[start_char_idx][0]  # 第一个字的开始时间
                end_time = timestamps[end_char_idx][1]     # 最后一个字的结束时间
                segment_timestamps.append((start_time, end_time))
            
            char_idx += char_count
            
        return segment_timestamps
    
    def cut_audio(self, audio_key: str, threshold: int = 6, 
                  extend_before_ms: int = 0, extend_after_ms: int = 0) -> List[Dict[str, Any]]:
        """
        切割指定音频
        
        Args:
            audio_key: 音频的key
            threshold: 每个片段的字数阈值
            extend_before_ms: 在片段开始前延伸的毫秒数
            extend_after_ms: 在片段结束后延伸的毫秒数
            
        Returns:
            切割信息列表
        """
        if audio_key not in self.datalist or audio_key not in self.timestamps:
            raise ValueError(f"找不到音频 {audio_key} 的数据")
        
        data = self.datalist[audio_key]
        timestamps = self.timestamps[audio_key]
        segment_label = data['segment_label']
        audio_path = data['sph']
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            print(f"警告: 音频文件不存在 {audio_path}")
            return []
        
        # 加载音频
        try:
            audio, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"错误: 无法加载音频文件 {audio_path}: {e}")
            return []
        
        # 计算每个词的字数
        word_char_counts = self._calculate_word_character_count(segment_label)
        
        # 根据阈值分组
        word_groups = self._group_words_by_threshold(word_char_counts, threshold)
        
        # 计算片段时间戳
        segment_timestamps = self._calculate_segment_timestamps(segment_label, timestamps, word_groups)
        
        # 切割音频并保存
        cut_info = []
        audio_duration_ms = len(audio) / sr * 1000  # 音频总时长（毫秒）
        
        # 获取完整文本（如果有的话）
        full_text = self.text_data.get(audio_key, "")
        
        for i, ((start_word, end_word), (start_time, end_time)) in enumerate(zip(word_groups, segment_timestamps)):
            # 应用延伸时间
            extended_start_time = max(0, start_time - extend_before_ms)
            extended_end_time = min(audio_duration_ms, end_time + extend_after_ms)
            
            # 转换时间戳为秒
            start_sec = start_time / 1000.0
            end_sec = end_time / 1000.0
            extended_start_sec = extended_start_time / 1000.0
            extended_end_sec = extended_end_time / 1000.0
            
            # 转换为样本索引（使用延伸后的时间）
            start_sample = int(extended_start_sec * sr)
            end_sample = int(extended_end_sec * sr)
            
            # 提取音频片段
            if start_sample < len(audio) and end_sample <= len(audio):
                audio_segment = audio[start_sample:end_sample]
                
                # 生成输出文件名
                output_filename = f"{audio_key}_segment_{i:03d}.wav"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # 保存音频片段
                sf.write(output_path, audio_segment, sr)
                
                # 收集切割信息
                words_in_segment = []
                char_count = 0
                for word_idx in range(start_word, end_word + 1):
                    word_chars = len(segment_label[word_idx])
                    words_in_segment.append({
                        'word_index': word_idx,
                        'char_count': word_chars
                    })
                    char_count += word_chars
                
                # 提取对应的文本片段
                text_segment = self._extract_text_segment(full_text, segment_label, start_word, end_word)
                
                segment_info = {
                    'segment_id': i,
                    'filename': output_filename,
                    'file_path': output_path,
                    'word_range': (start_word, end_word),
                    'words_info': words_in_segment,
                    'total_chars': char_count,
                    'text_segment': text_segment,  # 添加文本片段
                    'original_time_range_ms': (start_time, end_time),
                    'extended_time_range_ms': (extended_start_time, extended_end_time),
                    'original_time_range_sec': (start_sec, end_sec),
                    'extended_time_range_sec': (extended_start_sec, extended_end_sec),
                    'original_duration_sec': end_sec - start_sec,
                    'extended_duration_sec': extended_end_sec - extended_start_sec,
                    'extend_before_ms': extend_before_ms,
                    'extend_after_ms': extend_after_ms
                }
                cut_info.append(segment_info)
        
        return cut_info
    
    def cut_all_audio(self, threshold: int = 6, extend_before_ms: int = 0, extend_after_ms: int = 0) -> Dict[str, List[Dict[str, Any]]]:
        """
        切割所有音频
        
        Args:
            threshold: 每个片段的字数阈值
            extend_before_ms: 在片段开始前延伸的毫秒数
            extend_after_ms: 在片段结束后延伸的毫秒数
            
        Returns:
            所有音频的切割信息
        """
        all_cut_info = {}
        
        for idx, audio_key in enumerate(self.datalist.keys()):
            if idx % 100 == 0:
                print(f"正在处理音频: {idx}/{len(self.datalist.keys())}")
            # print(f"正在处理音频: {audio_key}")
            try:
                cut_info = self.cut_audio(audio_key, threshold, extend_before_ms, extend_after_ms)
                all_cut_info[audio_key] = cut_info
                # print(f"  成功切割为 {len(cut_info)} 个片段")
            except Exception as e:
               print(f"  处理失败: {e}")
               all_cut_info[audio_key] = []
        
        return all_cut_info
    
    def save_cut_info(self, cut_info: Dict[str, List[Dict[str, Any]]], info_filename: str = "cut_info.json"):
        """
        保存切割信息到JSON文件
        
        Args:
            cut_info: 切割信息
            info_filename: 信息文件名
        """
        info_path = os.path.join(self.output_dir, info_filename)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(cut_info, f, ensure_ascii=False, indent=2)
        print(f"切割信息已保存到: {info_path}")
    
    def save_text_segments(self, cut_info: Dict[str, List[Dict[str, Any]]], text_filename: str = "text_segments.txt"):
        """
        保存文本片段到文本文件
        
        Args:
            cut_info: 切割信息
            text_filename: 文本文件名
        """
        text_path = os.path.join(self.output_dir, text_filename)
        with open(text_path, 'w', encoding='utf-8') as f:
            for audio_key, segments in cut_info.items():
                for segment in segments:
                    segment_filename = segment['filename'].replace('.wav', '')
                    text_segment = segment.get('text_segment', '')
                    f.write(f"{segment_filename} {text_segment}\n")
        print(f"文本片段已保存到: {text_path}")


def main():
    """
    使用示例
    
    命令行参数:
    python script.py datalist_path timestamp_path text_path output_dir threshold [extend_before_ms] [extend_after_ms]
    
    例如:
    python script.py datalist.txt timestamp.txt text.txt output 6 100 100
    - 字数阈值: 6
    - 前延伸: 100毫秒
    - 后延伸: 100毫秒
    """
    # 配置文件路径
    #datalist_path = "datalist.txt"
    #timestamp_path = "timestamp.txt"
    #output_dir = "output_segments"
    datalist_path = sys.argv[1]
    timestamp_path = sys.argv[2]
    text_path = sys.argv[3]
    output_dir = sys.argv[4]
    threshold = int(sys.argv[5])
    extend_before_ms = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    extend_after_ms = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    
    # 创建切割工具
    tool = AudioSegmentationTool(datalist_path, timestamp_path, text_path, output_dir)
    
    # # 设置字数阈值
    # threshold = 6
    
    # 方式2: 切割所有音频
    print("\n开始切割所有音频...")
    print(f"配置参数: 字数阈值={threshold}, 前延伸={extend_before_ms}ms, 后延伸={extend_after_ms}ms")
    # if text_path:
    #     print(f"文本文件: {text_path}")
    # else:
    #     print("未提供文本文件，将不输出文本片段")
    
    all_cut_info = tool.cut_all_audio(threshold, extend_before_ms, extend_after_ms)
    
    # 保存切割信息
    tool.save_cut_info(all_cut_info)
    
    # 如果有文本数据，也保存文本片段
    if text_path and tool.text_data:
        tool.save_text_segments(all_cut_info)
    
    # 统计信息
    total_segments = sum(len(info) for info in all_cut_info.values())
    successful_audio = sum(1 for info in all_cut_info.values() if len(info) > 0)
    
    # 统计有文本片段的数量
    segments_with_text = 0
    if text_path:
        for segments in all_cut_info.values():
            for segment in segments:
                if segment.get('text_segment', ''):
                    segments_with_text += 1
    
    print(f"\n切割完成!")
    print(f"成功处理音频数: {successful_audio}")
    print(f"总切割片段数: {total_segments}")
    if text_path:
        print(f"包含文本的片段数: {segments_with_text}")
    print(f"字数阈值: {threshold}")
    print(f"时间延伸配置: 前{extend_before_ms}ms, 后{extend_after_ms}ms")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
