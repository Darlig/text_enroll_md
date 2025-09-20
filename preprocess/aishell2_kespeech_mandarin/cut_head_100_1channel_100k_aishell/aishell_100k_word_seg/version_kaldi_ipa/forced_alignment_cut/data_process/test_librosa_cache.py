#!/usr/bin/env python3
"""
测试librosa缓存行为
验证为什么需要预热而不能依赖第一次调用
"""

import time
import os
import sys
import psutil
import librosa
import json
import re
import ast

def test_librosa_consecutive_loads(audio_files, max_files=10):
    """测试连续加载多个音频文件的性能"""
    print(f"=== 测试连续加载 {min(len(audio_files), max_files)} 个音频文件 ===")
    
    times = []
    for i, audio_path in enumerate(audio_files[:max_files]):
        if not os.path.exists(audio_path):
            print(f"跳过不存在的文件: {audio_path}")
            continue
            
        print(f"加载第 {i+1} 个文件: {os.path.basename(audio_path)}")
        
        start_time = time.time()
        start_io = psutil.disk_io_counters()
        
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            
            end_time = time.time()
            end_io = psutil.disk_io_counters()
            
            duration = end_time - start_time
            read_bytes = end_io.read_bytes - start_io.read_bytes
            file_size = os.path.getsize(audio_path) / 1024  # KB
            
            times.append(duration)
            
            print(f"  耗时: {duration:.3f}s, 磁盘读取: {read_bytes/1024:.1f}KB, "
                  f"文件大小: {file_size:.1f}KB, 采样率: {sr}")
            
            # 如果这次加载时间异常长，分析原因
            if duration > 0.5:
                print(f"  ⚠️  异常耗时！可能原因:")
                print(f"     - librosa缓存失效")
                print(f"     - 文件格式特殊")
                print(f"     - 系统I/O压力")
                
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            continue
    
    if times:
        print(f"\n性能统计:")
        print(f"  平均耗时: {sum(times)/len(times):.3f}s")
        print(f"  最大耗时: {max(times):.3f}s")
        print(f"  最小耗时: {min(times):.3f}s")
        
        # 统计异常慢的文件
        slow_count = sum(1 for t in times if t > 0.1)
        print(f"  耗时>100ms的文件数: {slow_count}/{len(times)}")
        
        return times
    else:
        print("没有成功加载任何文件")
        return []

def test_with_without_warmup(audio_files, max_files=5):
    """比较有无预热的性能差异"""
    print(f"\n=== 比较有无预热的性能差异 ===")
    
    # 测试1：无预热
    print("\n1. 无预热测试:")
    # 重启Python进程模拟无预热状态（这里只是重新导入）
    import importlib
    importlib.reload(librosa)
    
    times_no_warmup = test_librosa_consecutive_loads(audio_files, max_files)
    
    # 测试2：有预热
    print("\n2. 有预热测试:")
    print("执行预热...")
    warmup_start = time.time()
    
    # 执行预热
    import numpy as np
    test_audio = np.random.randn(1024).astype(np.float32)
    _ = librosa.stft(test_audio)
    _ = librosa.feature.mfcc(test_audio, sr=16000, n_mfcc=13)
    
    # 用第一个文件进行实际预热
    if audio_files and os.path.exists(audio_files[0]):
        _ = librosa.load(audio_files[0], sr=None, duration=0.1)
    
    warmup_time = time.time() - warmup_start
    print(f"预热完成，耗时: {warmup_time:.3f}s")
    
    times_with_warmup = test_librosa_consecutive_loads(audio_files, max_files)
    
    # 比较结果
    if times_no_warmup and times_with_warmup:
        avg_no_warmup = sum(times_no_warmup) / len(times_no_warmup)
        avg_with_warmup = sum(times_with_warmup) / len(times_with_warmup)
        
        print(f"\n📊 性能对比:")
        print(f"  无预热平均耗时: {avg_no_warmup:.3f}s")
        print(f"  有预热平均耗时: {avg_with_warmup:.3f}s")
        print(f"  性能提升: {((avg_no_warmup - avg_with_warmup) / avg_no_warmup * 100):.1f}%")
        
        if avg_with_warmup < avg_no_warmup * 0.5:
            print("  ✅ 预热显著提升性能")
        elif avg_with_warmup < avg_no_warmup * 0.8:
            print("  🟡 预热有一定效果")
        else:
            print("  ❌ 预热效果不明显")

def load_audio_list_from_datalist(datalist_path, max_files=20):
    """从datalist文件中提取音频文件路径"""
    audio_files = []
    
    try:
        with open(datalist_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num >= max_files:
                    break
                    
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        audio_path = json_data.get('sph', '')
                        if audio_path:
                            audio_files.append(audio_path)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"读取datalist失败: {e}")
    
    print(f"从datalist中提取了 {len(audio_files)} 个音频文件路径")
    return audio_files

def test_memory_pressure_effect(audio_files, max_files=10):
    """测试内存压力对librosa缓存的影响"""
    print(f"\n=== 测试内存压力对缓存的影响 ===")
    
    times = []
    memory_usage = []
    
    for i, audio_path in enumerate(audio_files[:max_files]):
        if not os.path.exists(audio_path):
            continue
            
        # 记录加载前内存使用
        mem_before = psutil.virtual_memory().percent
        
        start_time = time.time()
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            end_time = time.time()
            
            # 记录加载后内存使用
            mem_after = psutil.virtual_memory().percent
            
            duration = end_time - start_time
            times.append(duration)
            memory_usage.append((mem_before, mem_after))
            
            print(f"文件 {i+1}: {duration:.3f}s, 内存: {mem_before:.1f}% -> {mem_after:.1f}%")
            
            # 模拟内存压力（创建大对象）
            if i == max_files // 2:
                print("  创建内存压力...")
                import numpy as np
                big_array = np.random.randn(50_000_000)  # 约400MB
                del big_array
                import gc
                gc.collect()
                print("  内存压力释放")
                
        except Exception as e:
            print(f"文件 {i+1} 加载失败: {e}")
    
    if times:
        # 分析前半部分和后半部分的性能差异
        mid = len(times) // 2
        first_half_avg = sum(times[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(times[mid:]) / (len(times) - mid) if len(times) > mid else 0
        
        print(f"\n内存压力影响分析:")
        print(f"  前半部分平均耗时: {first_half_avg:.3f}s")
        print(f"  后半部分平均耗时: {second_half_avg:.3f}s")
        
        if second_half_avg > first_half_avg * 1.5:
            print("  ⚠️  后半部分明显变慢，可能受内存压力影响")
        else:
            print("  ✅ 性能稳定，内存压力影响较小")

def main():
    if len(sys.argv) != 2:
        print("用法: python test_librosa_cache.py <datalist文件路径>")
        print("例如: python test_librosa_cache.py datalist.txt")
        sys.exit(1)
    
    datalist_path = sys.argv[1]
    
    if not os.path.exists(datalist_path):
        print(f"错误: datalist文件不存在: {datalist_path}")
        sys.exit(1)
    
    print("=== librosa缓存行为测试 ===")
    
    # 从datalist中提取音频文件
    audio_files = load_audio_list_from_datalist(datalist_path, max_files=20)
    
    if not audio_files:
        print("没有找到有效的音频文件")
        sys.exit(1)
    
    # 检查文件是否存在
    existing_files = [f for f in audio_files if os.path.exists(f)]
    print(f"找到 {len(existing_files)} 个存在的音频文件")
    
    if len(existing_files) < 3:
        print("音频文件太少，无法进行有效测试")
        sys.exit(1)
    
    # 测试1: 连续加载性能
    test_librosa_consecutive_loads(existing_files, max_files=10)
    
    # 测试2: 有无预热的对比
    test_with_without_warmup(existing_files, max_files=5)
    
    # 测试3: 内存压力影响
    test_memory_pressure_effect(existing_files, max_files=8)
    
    print(f"\n🎯 结论:")
    print(f"1. 如果连续加载中有多个文件耗时>100ms，说明需要预热")
    print(f"2. 如果预热能显著提升性能，说明预热有价值")
    print(f"3. 如果后半部分文件明显变慢，说明需要考虑内存管理")

if __name__ == "__main__":
    main()

