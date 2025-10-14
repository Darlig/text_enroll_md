"""
GPU优化的 Hugging Face Transformers Whisper embedding 提取工具
支持批量处理、内存优化、并行计算
"""

import torch
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
import librosa # 使用 librosa 进行音频加载
import math


# ==============================================================================
# 实用函数 (保持不变或略微调整)
# ==============================================================================

def generate_hierarchical_output_path(audio_path, dir_depth, output_dir):
    """
    根据原始音频路径生成层次化的输出路径
    """
    path_parts = audio_path.split('/')
    
    # 确定要保留的目录结构的起始点
    # 从后往前数 dir_depth 级目录，如果路径太短，就从头开始
    start_idx = max(0, len(path_parts) - 1 - dir_depth)
    
    # 构建相对路径 (排除文件名)
    relative_path = '/'.join(path_parts[start_idx:-1])
    output_subdir = os.path.join(output_dir, relative_path)
    
    # 确保输出目录存在
    os.makedirs(output_subdir, exist_ok=True)
    
    # 生成输出文件路径
    basename = os.path.basename(audio_path)
    # 去除文件扩展名
    base_id = '.'.join(basename.split('.')[:-1])
    filename = f"{base_id}.npy"
    output_path = os.path.join(output_subdir, filename)
    
    return output_path


def load_wav_scp(wav_scp_path: str) -> Dict[str, str]:
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


# ==============================================================================
# 核心提取逻辑 (Transformers 版本)
# ==============================================================================

def extract_embedding_batch(
    model: WhisperModel, 
    processor: WhisperProcessor,
    audio_paths: List[str], 
    uttids: List[str], 
    output_dir: str = "embeddings", 
    batch_size: int = 8, 
    preserve_structure: bool = True, 
    dir_depth: int = 3
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    批量提取 embedding（GPU优化版本，使用 Hugging Face Transformers）
    """
    results: List[Dict[str, Any]] = []
    success_count = 0
    error_count = 0
    device = model.device
    
    # 批量处理
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="批量提取 embedding"):
        batch_paths = audio_paths[i:i+batch_size]
        batch_uttids = uttids[i:i+batch_size]
        
        # 批量加载音频 (使用 librosa)
        raw_audios: List[np.ndarray] = []
        valid_indices: List[int] = []
        current_batch_uttids: List[str] = []
        current_batch_paths: List[str] = []
        
        # 阶段 1: 加载原始音频
        for j, audio_path in enumerate(batch_paths):
            try:
                # 使用 librosa 加载，保持与 processor 要求的 16kHz 一致
                audio_data, sr = librosa.load(audio_path, sr=16000)
                raw_audios.append(audio_data)
                valid_indices.append(j)
                current_batch_uttids.append(batch_uttids[j])
                current_batch_paths.append(audio_path)
            except Exception as e:
                results.append({
                    "uttid": batch_uttids[j],
                    "audio_path": audio_path,
                    "embedding_path": None,
                    "embedding_shape": None,
                    "success": False,
                    "error": f"加载音频失败: {str(e)}"
                })
                error_count += 1
        
        if not raw_audios:
            continue
            
        try:
            # 阶段 2: 预处理 (生成 Log-Mel 频谱图并自动补零/截断)
            # processor 内部执行了 '先补零/截断，后提特征' 的标准 Whisper 流程
            inputs = processor(
                raw_audios, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            # 将特征移动到设备上
            input_features = inputs.input_features.to(device)
            
            # 阶段 3: 批量提取特征
            with torch.no_grad():
                # WhisperModel 仅运行 Encoder 部分
                outputs = model.encoder(input_features)
                batch_features = outputs.last_hidden_state
                
                # 应用长度截断逻辑 (来自 test_transformer_whisper.py)
                batch_features_processed = []
                for k, audio_data in enumerate(raw_audios):
                    n_samples = audio_data.shape[0]
                    duration = n_samples / 16000
                    valid_feat_len = math.ceil(duration / 0.01)
                    valid_embed_len = math.ceil(valid_feat_len / 2)
                    
                    # 截断到有效长度
                    truncated_features = batch_features[k:k+1, :valid_embed_len, :]
                    batch_features_processed.append(truncated_features)
                
                # # 重新组合处理后的特征
                # print(f"batch_features_processed shape: {[ f.shape for f in batch_features_processed]}")
                # batch_features = torch.cat(batch_features_processed, dim=0)
                batch_features = batch_features_processed
            
            # 阶段 4: 保存批量结果
            for j, features in enumerate(batch_features):
                uttid = current_batch_uttids[j]
                audio_path = current_batch_paths[j]
                
                features_np = features.squeeze(0).cpu().numpy()
                
                # 生成输出路径
                if preserve_structure:
                    embedding_path = generate_hierarchical_output_path(
                        audio_path, dir_depth, output_dir
                    )
                else:
                    embedding_path = os.path.join(output_dir, f"{uttid}.npy")
                
                np.save(embedding_path, features_np)
                
                results.append({
                    "uttid": uttid,
                    "audio_path": audio_path,
                    "embedding_path": embedding_path,
                    "embedding_shape": features_np.shape,
                    "success": True,
                    "error": None
                })
                success_count += 1
                
        except Exception as e:
            # 批量处理失败，记录错误
            error_msg = f"批处理失败: {str(e)}"
            for j in valid_indices:
                 results.append({
                    "uttid": batch_uttids[j],
                    "audio_path": batch_paths[j],
                    "embedding_path": None,
                    "embedding_shape": None,
                    "success": False,
                    "error": error_msg
                })
            error_count += len(valid_indices)
            
    return results, success_count, error_count


def batch_extract_embeddings_optimized(
    model: WhisperModel, 
    processor: WhisperProcessor,
    wav_scp_path: str, 
    output_dir: str = "embeddings", 
    batch_size: int = 8, 
    num_workers: int = 4, # 注意：由于 GPU 批处理，此参数在当前实现中未被使用
    preserve_structure: bool = True, 
    dir_depth: int = 3
) -> List[Dict[str, Any]]:
    """
    GPU优化的批量提取 embedding
    """
    # 加载 wav.scp 文件
    wav_dict = load_wav_scp(wav_scp_path)
    print(f"加载了 {len(wav_dict)} 个音频文件")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    uttids = list(wav_dict.keys())
    audio_paths = list(wav_dict.values())
    
    print(f"使用批处理大小: {batch_size}")
    
    # 批量处理
    results, success_count, error_count = extract_embedding_batch(
        model=model,
        processor=processor,
        audio_paths=audio_paths,
        uttids=uttids,
        output_dir=output_dir,
        batch_size=batch_size,
        preserve_structure=preserve_structure,
        dir_depth=dir_depth
    )
    
    # 保存处理日志
    log_path = os.path.join(output_dir, "extraction_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Hugging Face Whisper Embedding 提取日志\n")
        f.write(f"模型: {model.config._name_or_path}\n")
        f.write(f"总文件数: {len(wav_dict)}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {error_count}\n")
        f.write(f"批处理大小: {batch_size}\n\n")
        
        for result in results:
            if result["success"]:
                f.write(f"✓ {result['uttid']}: {result['embedding_path']} ({result['embedding_shape']})\n")
            else:
                f.write(f"✗ {result['uttid']} ({result['audio_path']}): 失败原因 -> {result['error']}\n")
    
    print(f"\n批量处理完成:")
    print(f"总文件数: {len(wav_dict)}")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"日志保存到: {log_path}")
    
    return results


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description="GPU优化的批量提取 Hugging Face Whisper embedding")
    parser.add_argument("wav_scp", nargs='?', default=None, help="wav.scp 文件路径")
    parser.add_argument("--output_dir", "-o", default="embeddings_hf", help="输出目录")
    parser.add_argument("--model", default="openai/whisper-base", help="Whisper 模型名称或路径 (e.g., openai/whisper-large-v3)")
    parser.add_argument("--device", default="auto", help="设备类型 (auto/cpu/cuda)")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="批处理大小")
    parser.add_argument("--num_workers", "-w", type=int, default=4, help="工作进程数 (在当前实现中不影响 GPU 批处理)")
    parser.add_argument("--preserve_structure", action="store_true", default=True, help="保持原始目录结构")
    parser.add_argument("--single", help="处理单个音频文件（用于测试）")
    parser.add_argument("--dir_depth", type=int, default=3, help="保留原始目录深度")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    print(f"加载模型: {args.model}")
    
    # 1. 加载处理器 (用于预处理)
    try:
        processor = WhisperProcessor.from_pretrained(args.model)
        
        # 2. 加载模型 (仅 Encoder)
        # 仅加载 WhisperModel (Encoder部分)，不带解码器/语言头
        model = WhisperModel.from_pretrained(args.model).to(device)
    except Exception as e:
        print(f"加载模型或处理器失败: {e}")
        sys.exit(1)

    # ... (此处省略了 extract_embedding_single 的 transformers 版本，因为它在批量处理函数中已经被更好的错误处理取代)
    
    if args.single:
        # 处理单个文件（测试模式）
        print(f"测试模式: 处理单个文件 {args.single}")
        
        # 加载音频
        try:
            raw_audio, sr = librosa.load(args.single, sr=16000)
            inputs = processor(raw_audio, sampling_rate=16000, return_tensors="pt")
            
            input_features = inputs.input_features.to(device)
            
            with torch.no_grad():
                outputs = model.encoder(input_features)
                audio_embeddings = outputs.last_hidden_state
                
                # 应用长度截断逻辑 (来自 test_transformer_whisper.py)
                n_samples = raw_audio.shape[0]
                duration = n_samples / sr
                valid_feat_len = math.ceil(duration / 0.01)
                valid_embed_len = math.ceil(valid_feat_len / 2)
                
                audio_embeddings = audio_embeddings[:, :valid_embed_len, :]
                features = audio_embeddings.squeeze(0).cpu().numpy()
                
            # 保存
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            embedding_path = os.path.join(output_dir, f"{os.path.basename(args.single).split('.')[0]}.npy")
            np.save(embedding_path, features)
            
            print(f"✓ 成功处理: {args.single}")
            print(f"  Embedding 形状: {features.shape}")
            print(f"  保存路径: {embedding_path}")
            
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            
    elif args.wav_scp:
        # 批量处理
        if not os.path.exists(args.wav_scp):
            print(f"错误: wav.scp 文件不存在: {args.wav_scp}")
            sys.exit(1)
        
        print(f"开始GPU优化批量处理: {args.wav_scp}")
        batch_extract_embeddings_optimized(
            model=model,
            processor=processor,
            wav_scp_path=args.wav_scp,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            preserve_structure=args.preserve_structure,
            dir_depth=args.dir_depth
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()