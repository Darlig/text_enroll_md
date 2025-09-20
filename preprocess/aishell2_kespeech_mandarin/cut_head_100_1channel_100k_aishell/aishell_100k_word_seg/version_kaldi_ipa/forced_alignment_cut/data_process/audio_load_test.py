#!/usr/bin/env python3
"""
音频加载性能测试脚本
用于诊断 librosa.load() 的性能问题
"""

import time
import os
import sys
import psutil
import numpy as np

def test_librosa_load(audio_path: str, iterations: int = 5):
    """测试 librosa.load() 性能"""
    print(f"\n=== 测试 librosa.load() ===")
    print(f"文件: {audio_path}")
    print(f"文件大小: {os.path.getsize(audio_path) / 1024 / 1024:.2f} MB")
    
    try:
        import librosa
        
        times = []
        for i in range(iterations):
            print(f"第 {i+1} 次加载...")
            
            start_time = time.time()
            start_io = psutil.disk_io_counters()
            
            audio, sr = librosa.load(audio_path, sr=None)
            
            end_time = time.time()
            end_io = psutil.disk_io_counters()
            
            duration = end_time - start_time
            read_bytes = end_io.read_bytes - start_io.read_bytes
            
            times.append(duration)
            print(f"  耗时: {duration:.3f}s, 磁盘读取: {read_bytes/1024:.1f}KB, "
                  f"采样率: {sr}, 时长: {len(audio)/sr:.2f}s")
        
        avg_time = sum(times) / len(times)
        print(f"平均加载时间: {avg_time:.3f}s")
        
        if avg_time > 0.5:  # 超过500ms就算慢
            print("⚠️  librosa.load() 性能异常！")
            return False
        else:
            print("✅ librosa.load() 性能正常")
            return True
            
    except ImportError:
        print("❌ librosa 未安装")
        return False
    except Exception as e:
        print(f"❌ librosa.load() 失败: {e}")
        return False

def test_soundfile_load(audio_path: str, iterations: int = 5):
    """测试 soundfile.read() 性能"""
    print(f"\n=== 测试 soundfile.read() ===")
    
    try:
        import soundfile as sf
        
        times = []
        for i in range(iterations):
            print(f"第 {i+1} 次加载...")
            
            start_time = time.time()
            start_io = psutil.disk_io_counters()
            
            audio, sr = sf.read(audio_path)
            
            end_time = time.time()
            end_io = psutil.disk_io_counters()
            
            duration = end_time - start_time
            read_bytes = end_io.read_bytes - start_io.read_bytes
            
            times.append(duration)
            print(f"  耗时: {duration:.3f}s, 磁盘读取: {read_bytes/1024:.1f}KB, "
                  f"采样率: {sr}, 时长: {len(audio)/sr:.2f}s")
        
        avg_time = sum(times) / len(times)
        print(f"平均加载时间: {avg_time:.3f}s")
        
        if avg_time > 0.1:  # soundfile应该更快
            print("⚠️  soundfile.read() 性能异常！")
            return False
        else:
            print("✅ soundfile.read() 性能正常")
            return True
            
    except ImportError:
        print("❌ soundfile 未安装")
        return False
    except Exception as e:
        print(f"❌ soundfile.read() 失败: {e}")
        return False

def test_wave_load(audio_path: str, iterations: int = 5):
    """测试标准库 wave 性能"""
    print(f"\n=== 测试 wave 模块 ===")
    
    try:
        import wave
        
        times = []
        for i in range(iterations):
            print(f"第 {i+1} 次加载...")
            
            start_time = time.time()
            start_io = psutil.disk_io_counters()
            
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sr = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
            
            # 转换为numpy数组
            audio = np.frombuffer(frames, dtype=np.int16)
            if channels == 2:
                audio = audio.reshape(-1, 2)
            
            end_time = time.time()
            end_io = psutil.disk_io_counters()
            
            duration = end_time - start_time
            read_bytes = end_io.read_bytes - start_io.read_bytes
            
            times.append(duration)
            print(f"  耗时: {duration:.3f}s, 磁盘读取: {read_bytes/1024:.1f}KB, "
                  f"采样率: {sr}, 时长: {len(audio)/sr:.2f}s")
        
        avg_time = sum(times) / len(times)
        print(f"平均加载时间: {avg_time:.3f}s")
        
        if avg_time > 0.05:  # wave应该最快
            print("⚠️  wave 模块性能异常！")
            return False
        else:
            print("✅ wave 模块性能正常")
            return True
            
    except Exception as e:
        print(f"❌ wave 模块失败: {e}")
        return False

def test_file_io_performance(audio_path: str):
    """测试纯文件I/O性能"""
    print(f"\n=== 测试纯文件I/O性能 ===")
    
    try:
        file_size = os.path.getsize(audio_path)
        
        # 测试读取文件
        start_time = time.time()
        with open(audio_path, 'rb') as f:
            data = f.read()
        end_time = time.time()
        
        duration = end_time - start_time
        throughput = file_size / duration / 1024 / 1024  # MB/s
        
        print(f"文件大小: {file_size / 1024:.1f} KB")
        print(f"读取耗时: {duration:.3f}s")
        print(f"读取速度: {throughput:.1f} MB/s")
        
        if throughput < 50:  # 小于50MB/s就算慢
            print("⚠️  磁盘I/O性能异常！")
            return False
        else:
            print("✅ 磁盘I/O性能正常")
            return True
            
    except Exception as e:
        print(f"❌ 文件I/O测试失败: {e}")
        return False

def diagnose_system():
    """系统诊断"""
    print(f"\n=== 系统诊断 ===")
    
    # CPU信息
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
    
    print(f"CPU核心数: {cpu_count}")
    print(f"CPU使用率: {cpu_percent:.1f}%")
    print(f"负载平均值: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
    
    # 内存信息
    memory = psutil.virtual_memory()
    print(f"物理内存: {memory.total / 1024 / 1024 / 1024:.1f} GB")
    print(f"内存使用率: {memory.percent:.1f}%")
    print(f"可用内存: {memory.available / 1024 / 1024 / 1024:.1f} GB")
    
    # 磁盘信息
    try:
        disk_usage = psutil.disk_usage('/')
        print(f"磁盘总容量: {disk_usage.total / 1024 / 1024 / 1024:.1f} GB")
        print(f"磁盘使用率: {(disk_usage.used / disk_usage.total) * 100:.1f}%")
        print(f"磁盘可用: {disk_usage.free / 1024 / 1024 / 1024:.1f} GB")
    except:
        print("无法获取磁盘信息")
    
    # I/O统计
    try:
        io_counters = psutil.disk_io_counters()
        print(f"累计磁盘读取: {io_counters.read_bytes / 1024 / 1024 / 1024:.1f} GB")
        print(f"累计磁盘写入: {io_counters.write_bytes / 1024 / 1024 / 1024:.1f} GB")
    except:
        print("无法获取I/O统计")

def main():
    if len(sys.argv) != 2:
        print("用法: python audio_load_test.py <音频文件路径>")
        print("例如: python audio_load_test.py /path/to/audio.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在: {audio_path}")
        sys.exit(1)
    
    print("=== 音频加载性能诊断工具 ===")
    print(f"测试文件: {audio_path}")
    
    # 系统诊断
    diagnose_system()
    
    # 测试纯文件I/O
    io_ok = test_file_io_performance(audio_path)
    
    # 测试不同的音频加载方法
    wave_ok = test_wave_load(audio_path)
    sf_ok = test_soundfile_load(audio_path)
    librosa_ok = test_librosa_load(audio_path)
    
    # 总结
    print(f"\n=== 诊断结果 ===")
    print(f"文件I/O性能: {'✅ 正常' if io_ok else '❌ 异常'}")
    print(f"wave模块性能: {'✅ 正常' if wave_ok else '❌ 异常'}")
    print(f"soundfile性能: {'✅ 正常' if sf_ok else '❌ 异常'}")
    print(f"librosa性能: {'✅ 正常' if librosa_ok else '❌ 异常'}")
    
    if not librosa_ok:
        print(f"\n🔧 建议解决方案:")
        if sf_ok:
            print("1. 使用 soundfile 替代 librosa.load()")
        if wave_ok:
            print("2. 使用 wave 模块进行基础音频读取")
        if not io_ok:
            print("3. 检查磁盘性能和文件系统配置")
        print("4. 检查 librosa 依赖库版本")
        print("5. 尝试设置 NUMBA_CACHE_DIR 环境变量")

if __name__ == "__main__":
    main()

