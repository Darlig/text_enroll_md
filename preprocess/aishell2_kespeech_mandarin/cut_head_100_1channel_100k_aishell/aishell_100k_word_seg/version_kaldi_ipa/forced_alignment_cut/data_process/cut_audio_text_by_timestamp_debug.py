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
import psutil
import threading
from functools import wraps

def monitor_io(func):
    """装饰器：监控I/O操作时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        io_start = psutil.disk_io_counters()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        io_end = psutil.disk_io_counters()
        
        duration = end_time - start_time
        read_bytes = io_end.read_bytes - io_start.read_bytes
        write_bytes = io_end.write_bytes - io_start.write_bytes
        
        if duration > 0.1:  # 只记录耗时超过100ms的操作
            print(f"[I/O监控] {func.__name__}: {duration:.3f}s, "
                  f"读取: {read_bytes/1024/1024:.2f}MB, "
                  f"写入: {write_bytes/1024/1024:.2f}MB")
        
        return result
    return wrapper

class SystemMonitor:
    """系统资源监控器"""
    def __init__(self):
        self.monitoring = False
        self.stats = []
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # CPU和内存状态
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # I/O状态
                disk_io = psutil.disk_io_counters()
                
                # 进程状态
                current_process = psutil.Process()
                process_status = current_process.status()
                
                # 记录状态
                stat = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available': memory.available / 1024 / 1024 / 1024,  # GB
                    'disk_read': disk_io.read_bytes / 1024 / 1024,  # MB
                    'disk_write': disk_io.write_bytes / 1024 / 1024,  # MB
                    'process_status': process_status,
                    'load_avg': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
                }
                
                self.stats.append(stat)
                
                # 如果进程状态异常，记录详细信息
                if process_status in ['sleeping', 'disk-sleep']:
                    print(f"[监控警告] 进程状态: {process_status}, "
                          f"CPU: {cpu_percent:.1f}%, "
                          f"内存: {memory.percent:.1f}%, "
                          f"负载: {stat['load_avg']:.2f}")
                    
            except Exception as e:
                print(f"监控错误: {e}")
                
            time.sleep(1)
    
    def get_summary(self):
        """获取监控摘要"""
        if not self.stats:
            return "无监控数据"
            
        avg_cpu = sum(s['cpu_percent'] for s in self.stats) / len(self.stats)
        avg_memory = sum(s['memory_percent'] for s in self.stats) / len(self.stats)
        avg_load = sum(s['load_avg'] for s in self.stats) / len(self.stats)
        
        # 统计异常状态
        abnormal_states = [s for s in self.stats if s['process_status'] in ['sleeping', 'disk-sleep']]
        
        return (f"监控摘要 (共{len(self.stats)}个采样点):\n"
                f"  平均CPU使用率: {avg_cpu:.1f}%\n"
                f"  平均内存使用率: {avg_memory:.1f}%\n"
                f"  平均负载: {avg_load:.2f}\n"
                f"  异常状态次数: {len(abnormal_states)}")

class AudioSegmentationToolDebug:
    def __init__(self, datalist_path: str, timestamp_path: str, text_path: str = None, output_dir: str = "output_segments"):
        """调试版音频切割工具"""
        self.datalist_path = datalist_path
        self.timestamp_path = timestamp_path
        self.text_path = text_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 系统监控器
        self.monitor = SystemMonitor()
        
        print("=== 调试模式启动 ===")
        print(f"系统信息:")
        print(f"  CPU核心数: {psutil.cpu_count()}")
        print(f"  物理内存: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
        print(f"  磁盘信息: {self._get_disk_info()}")
        
        # 加载数据
        print("\n加载数据中...")
        start_time = time.time()
        self.datalist = self._load_datalist()
        self.timestamps = self._load_timestamps()
        self.text_data = self._load_text_data() if text_path else {}
        print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 预热librosa以避免首次加载延迟
        self._warmup_librosa()
    
    def _warmup_librosa(self):
        """预热librosa，避免首次加载延迟"""
        print("正在预热librosa...")
        try:
            import librosa
            import numpy as np
            
            # 创建一个小的测试音频数据
            test_audio = np.random.randn(1024).astype(np.float32)
            
            # 触发librosa的主要初始化过程
            _ = librosa.stft(test_audio)
            _ = librosa.feature.mfcc(test_audio, sr=16000, n_mfcc=13)
            
            print("librosa预热完成")
        except Exception as e:
            print(f"librosa预热失败: {e}")
        
    def _get_disk_info(self):
        """获取磁盘信息"""
        try:
            disk_usage = psutil.disk_usage('/')
            return f"总容量: {disk_usage.total / 1024 / 1024 / 1024:.1f} GB, " \
                   f"可用: {disk_usage.free / 1024 / 1024 / 1024:.1f} GB"
        except:
            return "无法获取磁盘信息"
    
    @monitor_io
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
        print(f"加载了 {len(datalist)} 条datalist记录")
        return datalist
    
    @monitor_io
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
        print(f"加载了 {len(timestamps)} 条timestamp记录")
        return timestamps
    
    @monitor_io
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
        print(f"加载了 {len(text_data)} 条文本记录")
        return text_data
    
    def _calculate_word_character_count(self, segment_label: List[List[List[int]]]) -> List[int]:
        """计算每个词的字数"""
        return [len(word) for word in segment_label]
    
    def _group_words_by_threshold(self, word_char_counts: List[int], threshold: int) -> List[Tuple[int, int]]:
        """根据阈值将词分组"""
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
    
    @monitor_io
    def _safe_audio_load(self, audio_path: str):
        """安全加载音频文件，带详细诊断"""
        print(f"[音频加载] 开始加载: {audio_path}")
        
        # 检查文件属性
        try:
            stat_info = os.stat(audio_path)
            file_size = stat_info.st_size / 1024 / 1024  # MB
            print(f"[音频加载] 文件大小: {file_size:.2f} MB")
        except Exception as e:
            print(f"[音频加载] 无法获取文件信息: {e}")
            
        # 记录加载前的状态
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            # 使用不同的加载方法进行测试
            print(f"[音频加载] 使用librosa加载...")
            audio, sr = librosa.load(audio_path, sr=None)
            
            load_time = time.time() - start_time
            end_memory = psutil.virtual_memory().percent
            memory_delta = end_memory - start_memory
            
            print(f"[音频加载] 加载完成: {load_time:.3f}s, "
                  f"采样率: {sr}, 时长: {len(audio)/sr:.2f}s, "
                  f"内存变化: {memory_delta:+.1f}%")
            
            return audio, sr
            
        except Exception as e:
            print(f"[音频加载] 加载失败: {e}")
            raise
    
    @monitor_io
    def _safe_audio_write(self, output_path: str, audio_segment: np.ndarray, sr: int):
        """安全写入音频文件"""
        print(f"[音频写入] 开始写入: {output_path}")
        start_time = time.time()
        
        try:
            sf.write(output_path, audio_segment, sr)
            write_time = time.time() - start_time
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"[音频写入] 写入完成: {write_time:.3f}s, 文件大小: {file_size:.1f} KB")
            
        except Exception as e:
            print(f"[音频写入] 写入失败: {e}")
            raise
    
    def cut_audio(self, audio_key: str, threshold: int = 6, 
                  extend_before_ms: int = 0, extend_after_ms: int = 0) -> List[Dict[str, Any]]:
        """切割指定音频 - 调试版"""
        print(f"\n=== 开始处理音频: {audio_key} ===")
        
        if audio_key not in self.datalist or audio_key not in self.timestamps:
            print(f"警告: 找不到音频 {audio_key} 的数据")
            return []
        
        data = self.datalist[audio_key]
        timestamps = self.timestamps[audio_key]
        segment_label = data['segment_label']
        audio_path = data['sph']
        
        # 检查音频文件
        if not os.path.exists(audio_path):
            print(f"警告: 音频文件不存在 {audio_path}")
            return []
        
        # 开始监控
        self.monitor.start_monitoring()
        
        try:
            # 加载音频
            audio, sr = self._safe_audio_load(audio_path)
            
            # 计算分组
            print(f"[处理] 计算词分组...")
            word_char_counts = self._calculate_word_character_count(segment_label)
            word_groups = self._group_words_by_threshold(word_char_counts, threshold)
            segment_timestamps = self._calculate_segment_timestamps(segment_label, timestamps, word_groups)
            
            print(f"[处理] 将切割为 {len(word_groups)} 个片段")
            
            # 获取完整文本
            full_text = self.text_data.get(audio_key, "")
            
            # 切割音频并保存
            cut_info = []
            audio_duration_ms = len(audio) / sr * 1000
            
            for i, ((start_word, end_word), (start_time, end_time)) in enumerate(zip(word_groups, segment_timestamps)):
                print(f"[处理] 处理片段 {i+1}/{len(word_groups)}")
                
                # 应用延伸时间
                extended_start_time = max(0, start_time - extend_before_ms)
                extended_end_time = min(audio_duration_ms, end_time + extend_after_ms)
                
                # 转换为样本索引
                start_sec = start_time / 1000.0
                end_sec = end_time / 1000.0
                extended_start_sec = extended_start_time / 1000.0
                extended_end_sec = extended_end_time / 1000.0
                
                start_sample = int(extended_start_sec * sr)
                end_sample = int(extended_end_sec * sr)
                
                if start_sample < len(audio) and end_sample <= len(audio):
                    audio_segment = audio[start_sample:end_sample]
                    
                    # 生成输出文件名
                    output_filename = f"{audio_key}_segment_{i:03d}.wav"
                    output_path = os.path.join(self.output_dir, output_filename)
                    
                    # 保存音频片段
                    self._safe_audio_write(output_path, audio_segment, sr)
                    
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
            
        except Exception as e:
            print(f"[错误] 处理音频失败: {e}")
            return []
        finally:
            # 停止监控并输出摘要
            self.monitor.stop_monitoring()
            print(f"\n{self.monitor.get_summary()}")
    
    def cut_all_audio_debug(self, threshold: int = 6, extend_before_ms: int = 0, 
                           extend_after_ms: int = 0, max_files: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """调试版批量处理 - 限制文件数量"""
        all_cut_info = {}
        audio_keys = list(self.datalist.keys())[:max_files]  # 只处理前几个文件
        
        print(f"\n=== 调试模式：只处理前 {len(audio_keys)} 个文件 ===")
        
        for idx, audio_key in enumerate(audio_keys):
            print(f"\n[{idx+1}/{len(audio_keys)}] 处理: {audio_key}")
            try:
                cut_info = self.cut_audio(audio_key, threshold, extend_before_ms, extend_after_ms)
                all_cut_info[audio_key] = cut_info
                print(f"成功切割为 {len(cut_info)} 个片段")
                
            except Exception as e:
                print(f"处理失败: {e}")
                all_cut_info[audio_key] = []
        
        return all_cut_info
    
    @monitor_io
    def save_cut_info(self, cut_info: Dict[str, List[Dict[str, Any]]], info_filename: str = "cut_info.json"):
        """保存切割信息"""
        info_path = os.path.join(self.output_dir, info_filename)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(cut_info, f, ensure_ascii=False, indent=2)
        print(f"切割信息已保存到: {info_path}")


def main():
    """调试版主函数"""
    if len(sys.argv) < 6:
        print("用法: python script.py datalist_path timestamp_path text_path output_dir threshold [extend_before_ms] [extend_after_ms]")
        sys.exit(1)
    
    datalist_path = sys.argv[1]
    timestamp_path = sys.argv[2]
    text_path = sys.argv[3]
    output_dir = sys.argv[4]
    threshold = int(sys.argv[5])
    extend_before_ms = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    extend_after_ms = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    
    print("=== 音频切割工具 - 调试模式 ===")
    print(f"配置参数:")
    print(f"  字数阈值: {threshold}")
    print(f"  前延伸: {extend_before_ms}ms")
    print(f"  后延伸: {extend_after_ms}ms")
    
    # 创建工具
    tool = AudioSegmentationToolDebug(datalist_path, timestamp_path, text_path, output_dir)
    
    # 调试处理
    start_time = time.time()
    all_cut_info = tool.cut_all_audio_debug(threshold, extend_before_ms, extend_after_ms)
    processing_time = time.time() - start_time
    
    # 保存结果
    tool.save_cut_info(all_cut_info)
    
    # 统计信息
    total_segments = sum(len(info) for info in all_cut_info.values())
    successful_audio = sum(1 for info in all_cut_info.values() if len(info) > 0)
    
    print(f"\n=== 调试处理完成 ===")
    print(f"总耗时: {processing_time:.2f}秒")
    print(f"成功处理音频数: {successful_audio}")
    print(f"总切割片段数: {total_segments}")
    
    # 性能建议
    if processing_time > 0 and successful_audio > 0:
        avg_time_per_file = processing_time / successful_audio
        print(f"平均每文件处理时间: {avg_time_per_file:.2f}秒")
        
        if avg_time_per_file > 5:
            print("\n⚠️  性能警告:")
            print("  - 每个文件处理时间过长")
            print("  - 检查磁盘性能和音频文件大小")
            print("  - 考虑使用SSD存储")


if __name__ == "__main__":
    main()
