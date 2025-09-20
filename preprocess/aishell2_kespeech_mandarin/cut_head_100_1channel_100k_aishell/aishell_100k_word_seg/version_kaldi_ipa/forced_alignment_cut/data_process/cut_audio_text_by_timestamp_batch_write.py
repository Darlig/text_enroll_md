import json
import os
from typing import List, Tuple, Dict, Any
import soundfile as sf
import numpy as np
import sys
import re
import ast
import time
from concurrent.futures import ThreadPoolExecutor
import threading

class AudioSegmentationToolBatchWrite:
    def __init__(self, datalist_path: str, timestamp_path: str, text_path: str = None, output_dir: str = "output_segments"):
        """
        批量写入版音频切割工具 - 优化磁盘I/O性能
        
        Args:
            datalist_path: 包含分词信息的datalist.txt文件路径
            timestamp_path: 包含时间戳信息的timestamp.txt文件路径
            text_path: 包含音频文本的文件路径
            output_dir: 输出目录
        """
        self.datalist_path = datalist_path
        self.timestamp_path = timestamp_path
        self.text_path = text_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 批量写入配置
        self.write_batch_size = 100  # 每批写入100个音频片段
        self.write_queue = []
        self.write_lock = threading.Lock()
        
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
    
    def _add_to_write_queue(self, audio_data: np.ndarray, sr: int, output_path: str):
        """添加音频数据到写入队列"""
        with self.write_lock:
            self.write_queue.append({
                'audio_data': audio_data.copy(),  # 复制数据避免引用问题
                'sr': sr,
                'output_path': output_path
            })
            
            # 如果队列满了，执行批量写入
            if len(self.write_queue) >= self.write_batch_size:
                self._flush_write_queue()
    
    def _flush_write_queue(self):
        """批量写入队列中的音频文件"""
        if not self.write_queue:
            return
        
        batch_to_write = self.write_queue.copy()
        self.write_queue.clear()
        
        # 使用线程池并行写入
        with ThreadPoolExecutor(max_workers=4) as executor:
            def write_single_file(item):
                try:
                    sf.write(item['output_path'], item['audio_data'], item['sr'])
                    return True
                except Exception as e:
                    print(f"写入失败 {item['output_path']}: {e}")
                    return False
            
            # 提交所有写入任务
            futures = [executor.submit(write_single_file, item) for item in batch_to_write]
            
            # 等待完成
            success_count = sum(1 for future in futures if future.result())
            
        # print(f"批量写入完成: {success_count}/{len(batch_to_write)} 个文件")
    
    def cut_audio(self, audio_key: str, threshold: int = 6, 
                  extend_before_ms: int = 0, extend_after_ms: int = 0) -> List[Dict[str, Any]]:
        """切割指定音频 - 使用批量写入"""
        if audio_key not in self.datalist or audio_key not in self.timestamps:
            raise ValueError(f"找不到音频 {audio_key} 的数据")
        
        data = self.datalist[audio_key]
        timestamps = self.timestamps[audio_key]
        segment_label = data['segment_label']
        audio_path = data['sph']
        
        if not os.path.exists(audio_path):
            print(f"警告: 音频文件不存在 {audio_path}")
            return []
        
        # 加载音频
        try:
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
        except Exception as e:
            print(f"错误: 无法加载音频文件 {audio_path}: {e}")
            return []
        
        # 计算分组和时间戳
        word_char_counts = self._calculate_word_character_count(segment_label)
        word_groups = self._group_words_by_threshold(word_char_counts, threshold)
        segment_timestamps = self._calculate_segment_timestamps(segment_label, timestamps, word_groups)
        
        # 处理音频片段
        cut_info = []
        audio_duration_ms = len(audio) / sr * 1000
        full_text = self.text_data.get(audio_key, "")
        
        for i, ((start_word, end_word), (start_time, end_time)) in enumerate(zip(word_groups, segment_timestamps)):
            # 时间计算
            extended_start_time = max(0, start_time - extend_before_ms)
            extended_end_time = min(audio_duration_ms, end_time + extend_after_ms)
            
            start_sec = start_time / 1000.0
            end_sec = end_time / 1000.0
            extended_start_sec = extended_start_time / 1000.0
            extended_end_sec = extended_end_time / 1000.0
            
            # 提取音频片段
            start_sample = int(extended_start_sec * sr)
            end_sample = int(extended_end_sec * sr)
            
            if start_sample < len(audio) and end_sample <= len(audio):
                audio_segment = audio[start_sample:end_sample]
                
                # 生成输出路径
                output_filename = f"{audio_key}_segment_{i:03d}.wav"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # 添加到写入队列而不是立即写入
                self._add_to_write_queue(audio_segment, sr, output_path)
                
                # 收集信息
                char_count = sum(len(segment_label[word_idx]) for word_idx in range(start_word, end_word + 1))
                text_segment = self._extract_text_segment(full_text, segment_label, start_word, end_word)
                
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
        
        return cut_info
    
    def cut_all_audio(self, threshold: int = 6, extend_before_ms: int = 0, extend_after_ms: int = 0) -> Dict[str, List[Dict[str, Any]]]:
        """切割所有音频 - 使用批量写入优化"""
        all_cut_info = {}
        total_files = len(self.datalist.keys())
        
        for idx, audio_key in enumerate(self.datalist.keys()):
            if idx % 100 == 0:
                print(f"正在处理音频: {idx}/{total_files} ({idx/total_files*100:.1f}%)")
            
            try:
                cut_info = self.cut_audio(audio_key, threshold, extend_before_ms, extend_after_ms)
                all_cut_info[audio_key] = cut_info
            except Exception as e:
               print(f"  处理失败 {audio_key}: {e}")
               all_cut_info[audio_key] = []
        
        # 处理完所有音频后，写入剩余的文件
        print("写入剩余的音频文件...")
        with self.write_lock:
            self._flush_write_queue()
        
        return all_cut_info
    
    def save_cut_info(self, cut_info: Dict[str, List[Dict[str, Any]]], info_filename: str = "cut_info.json"):
        """保存切割信息到JSON文件"""
        info_path = os.path.join(self.output_dir, info_filename)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(cut_info, f, ensure_ascii=False, indent=2)
        print(f"切割信息已保存到: {info_path}")
    
    def save_text_segments(self, cut_info: Dict[str, List[Dict[str, Any]]], text_filename: str = "text_segments.txt"):
        """保存文本片段到文本文件"""
        text_path = os.path.join(self.output_dir, text_filename)
        with open(text_path, 'w', encoding='utf-8') as f:
            for audio_key, segments in cut_info.items():
                for segment in segments:
                    segment_filename = segment['filename'].replace('.wav', '')
                    text_segment = segment.get('text_segment', '')
                    f.write(f"{segment_filename} {text_segment}\n")
        print(f"文本片段已保存到: {text_path}")


def main():
    """批量写入版主函数"""
    datalist_path = sys.argv[1]
    timestamp_path = sys.argv[2]
    text_path = sys.argv[3]
    output_dir = sys.argv[4]
    threshold = int(sys.argv[5])
    extend_before_ms = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    extend_after_ms = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    
    print("=== 批量写入优化版音频切割工具 ===")
    print(f"配置参数:")
    print(f"  字数阈值: {threshold}")
    print(f"  前延伸: {extend_before_ms}ms")
    print(f"  后延伸: {extend_after_ms}ms")
    print(f"  批量写入大小: 100个文件/批")
    
    # 创建切割工具
    tool = AudioSegmentationToolBatchWrite(datalist_path, timestamp_path, text_path, output_dir)
    
    # 切割所有音频
    print("\n开始切割所有音频...")
    start_time = time.time()
    all_cut_info = tool.cut_all_audio(threshold, extend_before_ms, extend_after_ms)
    processing_time = time.time() - start_time
    
    # 保存切割信息
    tool.save_cut_info(all_cut_info)
    
    # 保存文本片段
    if text_path and tool.text_data:
        tool.save_text_segments(all_cut_info)
    
    # 统计信息
    total_segments = sum(len(info) for info in all_cut_info.values())
    successful_audio = sum(1 for info in all_cut_info.values() if len(info) > 0)
    
    print(f"\n切割完成!")
    print(f"总耗时: {processing_time:.2f}秒")
    print(f"成功处理音频数: {successful_audio}")
    print(f"总切割片段数: {total_segments}")
    if successful_audio > 0:
        print(f"平均处理速度: {successful_audio/processing_time:.2f} 文件/秒")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
