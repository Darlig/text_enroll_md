"""
高吞吐量 Whisper embedding 提取工具
针对大量数据优化，支持多进程、内存优化、异步I/O
"""

import torch
import whisper
import numpy as np
import os
import argparse
import sys
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from whisper.decoding import DecodingOptions
from whisper.audio import N_FRAMES


def extract_embedding_worker(args):
    """
    工作进程函数，用于多进程处理
    """
    uttid, audio_path, output_dir, model_name, device = args
    
    try:
        # 在工作进程中加载模型（避免模型序列化问题）
        model = whisper.load_model(model_name, device=device)
        
        # 加载音频
        audio_data = whisper.load_audio(audio_path)
        
        # 生成梅尔频谱图
        mel = whisper.log_mel_spectrogram(audio_data, model.dims.n_mels)
        mel = whisper.pad_or_trim(mel, N_FRAMES)
        
        # 移动到设备
        mel = mel.to(model.device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        
        # 提取特征
        with torch.no_grad():
            audio_features = model.encoder(mel)
        
        # 保存embedding
        embedding_path = os.path.join(output_dir, f"{uttid}.npy")
        np.save(embedding_path, audio_features.cpu().numpy())
        
        return {
            "uttid": uttid,
            "audio_path": audio_path,
            "embedding_path": embedding_path,
            "embedding_shape": audio_features.shape,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "uttid": uttid,
            "audio_path": audio_path,
            "embedding_path": None,
            "embedding_shape": None,
            "success": False,
            "error": str(e)
        }


def load_wav_scp(wav_scp_path):
    """加载 wav.scp 文件"""
    wav_dict = {}
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    uttid, audio_path = parts
                    wav_dict[uttid] = audio_path
    return wav_dict


def batch_extract_embeddings_high_throughput(wav_scp_path, output_dir="embeddings", 
                                           model_name="base", device="auto", 
                                           num_processes=None, chunk_size=100):
    """
    高吞吐量批量提取 embedding
    
    Parameters
    ----------
    wav_scp_path : str
        wav.scp 文件路径
    output_dir : str
        输出目录
    model_name : str
        模型名称
    device : str
        设备类型
    num_processes : int
        进程数，None表示自动设置
    chunk_size : int
        每个进程处理的文件数
    """
    # 设置设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设置进程数
    if num_processes is None:
        if device == "cuda":
            num_processes = min(4, mp.cpu_count())  # GPU模式限制进程数
        else:
            num_processes = mp.cpu_count()
    
    print(f"使用设备: {device}")
    print(f"使用进程数: {num_processes}")
    print(f"模型: {model_name}")
    
    # 加载 wav.scp 文件
    wav_dict = load_wav_scp(wav_scp_path)
    print(f"加载了 {len(wav_dict)} 个音频文件")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备任务列表
    tasks = []
    for uttid, audio_path in wav_dict.items():
        tasks.append((uttid, audio_path, output_dir, model_name, device))
    
    # 分批处理
    results = []
    success_count = 0
    error_count = 0
    
    # 将任务分批
    task_chunks = [tasks[i:i+chunk_size] for i in range(0, len(tasks), chunk_size)]
    
    print(f"分 {len(task_chunks)} 批处理，每批 {chunk_size} 个文件")
    
    start_time = time.time()
    
    # 使用进程池处理
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for chunk_idx, task_chunk in enumerate(task_chunks):
            print(f"处理第 {chunk_idx+1}/{len(task_chunks)} 批...")
            
            # 提交任务
            future_to_task = {executor.submit(extract_embedding_worker, task): task for task in task_chunk}
            
            # 收集结果
            for future in tqdm(future_to_task, desc=f"批次 {chunk_idx+1}"):
                try:
                    result = future.result(timeout=300)  # 5分钟超时
                    results.append(result)
                    
                    if result["success"]:
                        success_count += 1
                    else:
                        error_count += 1
                        print(f"处理失败 {result['uttid']}: {result['error']}")
                        
                except Exception as e:
                    task = future_to_task[future]
                    error_count += 1
                    print(f"任务超时或失败 {task[0]}: {e}")
    
    total_time = time.time() - start_time
    
    # 保存处理日志
    log_path = os.path.join(output_dir, "extraction_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"高吞吐量批量提取 embedding 日志\n")
        f.write(f"总文件数: {len(wav_dict)}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {error_count}\n")
        f.write(f"总耗时: {total_time:.2f}秒\n")
        f.write(f"平均每文件: {total_time/len(wav_dict):.3f}秒\n")
        f.write(f"吞吐量: {len(wav_dict)/total_time:.2f} 文件/秒\n")
        f.write(f"进程数: {num_processes}\n")
        f.write(f"设备: {device}\n\n")
        
        for result in results:
            if result["success"]:
                f.write(f"✓ {result['uttid']}: {result['embedding_path']}\n")
            else:
                f.write(f"✗ {result['uttid']}: {result['error']}\n")
    
    print(f"\n高吞吐量批量处理完成:")
    print(f"总文件数: {len(wav_dict)}")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每文件: {total_time/len(wav_dict):.3f}秒")
    print(f"吞吐量: {len(wav_dict)/total_time:.2f} 文件/秒")
    print(f"日志保存到: {log_path}")
    
    return results


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description="高吞吐量批量提取 Whisper embedding")
    parser.add_argument("wav_scp", help="wav.scp 文件路径")
    parser.add_argument("--output_dir", "-o", default="embeddings", help="输出目录")
    parser.add_argument("--model", default="base", help="Whisper 模型大小")
    parser.add_argument("--device", default="auto", help="设备类型 (auto/cpu/cuda)")
    parser.add_argument("--num_processes", "-p", type=int, help="进程数")
    parser.add_argument("--chunk_size", "-c", type=int, default=100, help="每批处理文件数")
    parser.add_argument("--single", help="处理单个音频文件（用于测试）")
    
    args = parser.parse_args()
    
    if args.single:
        # 处理单个文件（测试模式）
        print(f"测试模式: 处理单个文件 {args.single}")
        
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(args.model, device=device)
        
        # 加载音频
        audio_data = whisper.load_audio(args.single)
        mel = whisper.log_mel_spectrogram(audio_data, model.dims.n_mels)
        mel = whisper.pad_or_trim(mel, N_FRAMES)
        mel = mel.to(model.device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        
        with torch.no_grad():
            audio_features = model.encoder(mel)
        
        # 保存embedding
        os.makedirs(args.output_dir, exist_ok=True)
        embedding_path = os.path.join(args.output_dir, "test.npy")
        np.save(embedding_path, audio_features.cpu().numpy())
        
        print(f"✓ 成功处理")
        print(f"  Embedding 形状: {audio_features.shape}")
        print(f"  保存路径: {embedding_path}")
    else:
        # 批量处理
        if not os.path.exists(args.wav_scp):
            print(f"错误: wav.scp 文件不存在: {args.wav_scp}")
            sys.exit(1)
        
        print(f"开始高吞吐量批量处理: {args.wav_scp}")
        results = batch_extract_embeddings_high_throughput(
            wav_scp_path=args.wav_scp,
            output_dir=args.output_dir,
            model_name=args.model,
            device=args.device,
            num_processes=args.num_processes,
            chunk_size=args.chunk_size
        )


if __name__ == "__main__":
    main()
