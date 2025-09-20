import json
import os
from typing import List, Tuple, Dict, Any
import librosa
import soundfile as sf
import numpy as np
import sys
import re
import ast
import time
import fcntl
import hashlib
import tempfile
import shutil
from pathlib import Path

class AudioSegmentationToolSafe:
    def __init__(self, datalist_path: str, timestamp_path: str, text_path: str = None, output_dir: str = "output_segments"):
        """
        安全版音频切割工具 - 避免多进程竞争
        
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
        
        # 为当前实例创建唯一的临时工作目录
        self.instance_id = hashlib.md5(f"{datalist_path}_{timestamp_path}_{time.time()}".encode()).hexdigest()[:8]
        self.temp_dir = os.path.join(tempfile.gettempdir(), f"audio_cut_{self.instance_id}")
        
        # 创建目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 创建锁文件防止多实例同时处理相同数据
        self.lock_file = os.path.join(output_dir, f".processing_lock_{self.instance_id}")
        
        print(f"实例ID: {self.instance_id}")
        print(f"临时目录: {self.temp_dir}")
        
        # 加载数据
        print("加载数据中...")
        start_time = time.time()
        self.datalist = self._load_datalist()
        self.timestamps = self._load_timestamps()
        self.text_data = self._load_text_data() if text_path else {}
        print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 设置资源限制
        self._setup_resource_limits()
        
    def _setup_resource_limits(self):
        """设置资源限制以避免系统过载"""
        try:
            import resource
            # 限制最大内存使用（可选）
            # resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024, -1))  # 8GB
            
            # 限制最大打开文件数
            resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 4096))
        except ImportError:
            pass  # Windows系统没有resource模块
        
    def _load_datalist(self) -> Dict[str, Any]:
        """加载分词信息"""
        datalist = {}
        try:
            with open(self.datalist_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            json_data = json.loads(line)
                            datalist[json_data['key']] = json_data
                        except json.JSONDecodeError as e:
                            print(f"警告: 第{line_num}行JSON解析失败: {e}")
                            continue
        except Exception as e:
            print(f"错误: 无法读取datalist文件 {self.datalist_path}: {e}")
            
        print(f"成功加载 {len(datalist)} 条datalist记录")
        return datalist
    
    def _load_timestamps(self) -> Dict[str, List[List[int]]]:
        """加载时间戳信息"""
        timestamps = {}
        try:
            with open(self.timestamp_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            parts = re.split(r'\s+', line, maxsplit=1)
                            if len(parts) != 2:
                                continue
                            key = parts[0]
                            timestamp_data = parts[1]
                            timestamps[key] = ast.literal_eval(timestamp_data)
                        except (ValueError, SyntaxError) as e:
                            print(f"警告: 第{line_num}行时间戳解析失败: {e}")
                            continue
        except Exception as e:
            print(f"错误: 无法读取timestamp文件 {self.timestamp_path}: {e}")
            
        print(f"成功加载 {len(timestamps)} 条timestamp记录")
        return timestamps
    
    def _load_text_data(self) -> Dict[str, str]:
        """加载音频文本数据"""
        text_data = {}
        if not self.text_path:
            return text_data
            
        try:
            with open(self.text_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            uttid, text_content = parts
                            text_data[uttid] = text_content
        except Exception as e:
            print(f"错误: 无法读取文本文件 {self.text_path}: {e}")
            
        print(f"成功加载 {len(text_data)} 条文本记录")
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
    
    def _safe_audio_load(self, audio_path: str, max_retries: int = 3):
        """安全的音频加载，带重试机制"""
        for attempt in range(max_retries):
            try:
                # 检查文件是否存在且可读
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"音频文件不存在: {audio_path}")
                
                if not os.access(audio_path, os.R_OK):
                    raise PermissionError(f"音频文件无读取权限: {audio_path}")
                
                # 加载音频
                audio, sr = librosa.load(audio_path, sr=None)
                return audio, sr
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"音频加载失败，重试中 ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(0.1 * (attempt + 1))  # 指数退避
                    continue
                else:
                    raise e
    
    def _safe_audio_write(self, output_path: str, audio_segment: np.ndarray, sr: int, max_retries: int = 3):
        """安全的音频写入，避免文件竞争"""
        # 先写入临时文件，然后原子性移动
        temp_path = os.path.join(self.temp_dir, os.path.basename(output_path))
        
        for attempt in range(max_retries):
            try:
                # 写入临时文件
                sf.write(temp_path, audio_segment, sr)
                
                # 原子性移动到最终位置
                shutil.move(temp_path, output_path)
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"音频写入失败，重试中 ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    print(f"音频写入最终失败: {e}")
                    return False
    
    def cut_audio(self, audio_key: str, threshold: int = 6, 
                  extend_before_ms: int = 0, extend_after_ms: int = 0) -> List[Dict[str, Any]]:
        """切割指定音频"""
        if audio_key not in self.datalist or audio_key not in self.timestamps:
            print(f"警告: 找不到音频 {audio_key} 的数据")
            return []
        
        data = self.datalist[audio_key]
        timestamps = self.timestamps[audio_key]
        segment_label = data['segment_label']
        audio_path = data['sph']
        
        try:
            # 安全加载音频
            audio, sr = self._safe_audio_load(audio_path)
        except Exception as e:
            print(f"错误: 无法加载音频文件 {audio_path}: {e}")
            return []
        
        # 计算分组
        word_char_counts = self._calculate_word_character_count(segment_label)
        word_groups = self._group_words_by_threshold(word_char_counts, threshold)
        segment_timestamps = self._calculate_segment_timestamps(segment_label, timestamps, word_groups)
        
        # 获取完整文本
        full_text = self.text_data.get(audio_key, "")
        
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
                output_path = os.path.join(self.output_dir, output_filename)
                
                # 安全保存音频片段
                if self._safe_audio_write(output_path, audio_segment, sr):
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
    
    def cut_all_audio_safe(self, threshold: int = 6, extend_before_ms: int = 0, 
                          extend_after_ms: int = 0, batch_size: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        安全的批量音频处理，避免资源竞争
        
        Args:
            threshold: 每个片段的字数阈值
            extend_before_ms: 在片段开始前延伸的毫秒数
            extend_after_ms: 在片段结束后延伸的毫秒数
            batch_size: 批处理大小，控制内存使用
        """
        all_cut_info = {}
        audio_keys = list(self.datalist.keys())
        total_files = len(audio_keys)
        
        print(f"开始处理 {total_files} 个音频文件，批大小: {batch_size}")
        
        # 分批处理
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_keys = audio_keys[batch_start:batch_end]
            
            print(f"处理批次 {batch_start//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}: "
                  f"文件 {batch_start + 1}-{batch_end}")
            
            batch_start_time = time.time()
            
            for i, audio_key in enumerate(batch_keys):
                try:
                    cut_info = self.cut_audio(audio_key, threshold, extend_before_ms, extend_after_ms)
                    all_cut_info[audio_key] = cut_info
                    
                    if (i + 1) % 5 == 0:
                        print(f"  已处理: {i + 1}/{len(batch_keys)}")
                        
                except Exception as e:
                    print(f"  处理失败 {audio_key}: {e}")
                    all_cut_info[audio_key] = []
            
            batch_time = time.time() - batch_start_time
            print(f"批次完成，耗时: {batch_time:.2f}秒")
            
            # 强制垃圾回收，释放内存
            import gc
            gc.collect()
            
            # 小延迟，减少系统压力
            time.sleep(0.1)
        
        return all_cut_info
    
    def save_cut_info_safe(self, cut_info: Dict[str, List[Dict[str, Any]]], 
                          info_filename: str = "cut_info.json"):
        """安全保存切割信息"""
        temp_path = os.path.join(self.temp_dir, info_filename)
        final_path = os.path.join(self.output_dir, info_filename)
        
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(cut_info, f, ensure_ascii=False, indent=2)
            
            shutil.move(temp_path, final_path)
            print(f"切割信息已保存到: {final_path}")
            
        except Exception as e:
            print(f"保存切割信息失败: {e}")
    
    def save_text_segments_safe(self, cut_info: Dict[str, List[Dict[str, Any]]], 
                               text_filename: str = "text_segments.txt"):
        """安全保存文本片段"""
        temp_path = os.path.join(self.temp_dir, text_filename)
        final_path = os.path.join(self.output_dir, text_filename)
        
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                for audio_key, segments in cut_info.items():
                    for segment in segments:
                        segment_filename = segment['filename'].replace('.wav', '')
                        text_segment = segment.get('text_segment', '')
                        f.write(f"{segment_filename} {text_segment}\n")
            
            shutil.move(temp_path, final_path)
            print(f"文本片段已保存到: {final_path}")
            
        except Exception as e:
            print(f"保存文本片段失败: {e}")
    
    def cleanup(self):
        """清理临时文件"""
        try:
            shutil.rmtree(self.temp_dir)
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except Exception as e:
            print(f"清理临时文件失败: {e}")


def main():
    """安全版主函数"""
    if len(sys.argv) < 6:
        print("用法: python script.py datalist_path timestamp_path text_path output_dir threshold [extend_before_ms] [extend_after_ms] [batch_size]")
        sys.exit(1)
    
    datalist_path = sys.argv[1]
    timestamp_path = sys.argv[2]
    text_path = sys.argv[3]
    output_dir = sys.argv[4]
    threshold = int(sys.argv[5])
    extend_before_ms = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    extend_after_ms = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    batch_size = int(sys.argv[8]) if len(sys.argv) > 8 else 10
    
    print("=== 安全音频切割工具 ===")
    print(f"配置参数:")
    print(f"  字数阈值: {threshold}")
    print(f"  前延伸: {extend_before_ms}ms")
    print(f"  后延伸: {extend_after_ms}ms")
    print(f"  批大小: {batch_size}")
    print(f"  输出目录: {output_dir}")
    
    # 创建工具
    tool = None
    try:
        tool = AudioSegmentationToolSafe(datalist_path, timestamp_path, text_path, output_dir)
        
        # 处理音频
        start_time = time.time()
        all_cut_info = tool.cut_all_audio_safe(threshold, extend_before_ms, extend_after_ms, batch_size)
        processing_time = time.time() - start_time
        
        # 保存结果
        print("\n保存结果中...")
        tool.save_cut_info_safe(all_cut_info)
        if text_path and tool.text_data:
            tool.save_text_segments_safe(all_cut_info)
        
        # 统计信息
        total_segments = sum(len(info) for info in all_cut_info.values())
        successful_audio = sum(1 for info in all_cut_info.values() if len(info) > 0)
        segments_with_text = sum(1 for segments in all_cut_info.values() 
                               for segment in segments if segment.get('text_segment', ''))
        
        print(f"\n=== 处理完成 ===")
        print(f"总耗时: {processing_time:.2f}秒")
        print(f"成功处理音频数: {successful_audio}")
        print(f"总切割片段数: {total_segments}")
        print(f"包含文本的片段数: {segments_with_text}")
        if successful_audio > 0:
            print(f"平均处理速度: {successful_audio/processing_time:.2f} 文件/秒")
        
    except KeyboardInterrupt:
        print("\n收到中断信号，正在清理...")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
    finally:
        if tool:
            tool.cleanup()


if __name__ == "__main__":
    main()

