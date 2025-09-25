"""
GPU优化的 Whisper embedding 提取工具
支持批量处理、内存优化、并行计算
"""

import torch
import whisper
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from whisper.decoding import DecodingOptions
from whisper.audio import N_FRAMES


def generate_hierarchical_output_path(audio_path, dir_depth, output_dir):
    """
    根据原始音频路径生成层次化的输出路径
    
    Parameters
    ----------
    audio_path : str
        原始音频文件路径
    dir_depth : int
        保留原始目录深度
    output_dir : str
        输出根目录
        
    Returns
    -------
    str
        层次化的输出路径
    """
    # 从音频路径中提取目录结构
    # 例如: /work104/weiyang/data/TIMIT/data/TEST/DR1/FAKS0/SA1.WAV
    # 提取: TEST/DR1/FAKS0/
    path_parts = audio_path.split('/')
    
    # 找到数据根目录（通常包含TEST或TRAIN等标识）
    data_root_idx = -1
    # for i, part in enumerate(path_parts):
    #     if part in ['TEST', 'TRAIN', 'data']:
    #         data_root_idx = i
    #         break
    
    if data_root_idx == -1:
        # 如果找不到标准的数据根目录，使用文件名前的最后两级目录
        data_root_idx = max(0, len(path_parts) - dir_depth)
    
    # 构建相对路径
    # if data_root_idx < len(path_parts) - 1:
    relative_path = '/'.join(path_parts[data_root_idx:-1])  # 排除文件名
    output_subdir = os.path.join(output_dir, relative_path)
    # else:
    #     # 如果路径太短，使用uttid的前几个字符作为子目录
    #     subdir = uttid[:8] if len(uttid) >= 8 else uttid
    #     output_subdir = os.path.join(output_dir, subdir)
    
    # 确保输出目录存在
    os.makedirs(output_subdir, exist_ok=True)
    
    # 生成输出文件路径
    basename = os.path.basename(audio_path)
    # 去除文件扩展名
    base_id = '.'.join(basename.split('.')[:-1])
    filename = f"{base_id}.npy"
    output_path = os.path.join(output_subdir, filename)
    
    return output_path


def extract_embedding_batch(model, audio_paths, uttids, output_dir="embeddings", batch_size=8, preserve_structure=True, dir_depth=3):
    """
    批量提取 embedding（GPU优化版本）
    
    Parameters
    ----------
    model : Whisper
        Whisper 模型实例
    audio_paths : list
        音频文件路径列表
    uttids : list
        音频ID列表
    output_dir : str
        输出目录
    batch_size : int
        批处理大小
    preserve_structure : bool
        是否保持原始目录结构
        
    Returns
    -------
    list
        处理结果列表
    """
    results = []
    success_count = 0
    error_count = 0
    
    # 批量处理
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="批量提取 embedding"):
        batch_paths = audio_paths[i:i+batch_size]
        batch_uttids = uttids[i:i+batch_size]
        
        try:
            # 批量加载音频
            batch_audios = []
            valid_indices = []
            
            for j, audio_path in enumerate(batch_paths):
                try:
                    audio_data = whisper.load_audio(audio_path)
                    batch_audios.append(audio_data)
                    valid_indices.append(j)
                except Exception as e:
                    results.append({
                        "uttid": batch_uttids[j],
                        "audio_path": audio_path,
                        "embedding_path": None,
                        "embedding_shape": None,
                        "success": False,
                        "error": str(e)
                    })
                    error_count += 1
            
            if not batch_audios:
                continue
                
            # 批量生成梅尔频谱图
            batch_mels = []
            for audio_data in batch_audios:
                mel = whisper.log_mel_spectrogram(audio_data, model.dims.n_mels)
                mel = whisper.pad_or_trim(mel, N_FRAMES)
                batch_mels.append(mel)
            
            # 堆叠为批量张量
            batch_mel = torch.stack(batch_mels).to(model.device)
            
            # 批量提取特征
            with torch.no_grad():
                batch_features = model.encoder(batch_mel)
            
            # 保存批量结果
            for j, idx in enumerate(valid_indices):
                uttid = batch_uttids[idx]
                features = batch_features[j].cpu().numpy()
                
                # 生成输出路径
                if preserve_structure:
                    embedding_path = generate_hierarchical_output_path(
                        batch_paths[idx], dir_depth, output_dir
                    )
                else:
                    embedding_path = os.path.join(output_dir, f"{uttid}.npy")
                
                np.save(embedding_path, features)
                
                results.append({
                    "uttid": uttid,
                    "audio_path": batch_paths[idx],
                    "embedding_path": embedding_path,
                    "embedding_shape": features.shape,
                    "success": True,
                    "error": None
                })
                success_count += 1
                
        except Exception as e:
            # 批量处理失败，回退到单个处理
            for j, (audio_path, uttid) in enumerate(zip(batch_paths, batch_uttids)):
                try:
                    result = extract_embedding_single(model, audio_path, uttid, output_dir, preserve_structure)
                    results.append(result)
                    if result["success"]:
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as single_e:
                    results.append({
                        "uttid": uttid,
                        "audio_path": audio_path,
                        "embedding_path": None,
                        "embedding_shape": None,
                        "success": False,
                        "error": str(single_e)
                    })
                    error_count += 1
    
    return results, success_count, error_count


def extract_embedding_single(model, audio_path, uttid, output_dir, preserve_structure=True):
    """
    单个音频文件提取 embedding
    """
    try:
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
        
        # 生成输出路径
        if preserve_structure:
            embedding_path = generate_hierarchical_output_path(audio_path, uttid, output_dir)
        else:
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


def batch_extract_embeddings_optimized(model, wav_scp_path, output_dir="embeddings", batch_size=8, num_workers=4, preserve_structure=True, dir_depth=3):
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
    print(f"使用工作进程数: {num_workers}")
    
    # 批量处理
    results, success_count, error_count = extract_embedding_batch(
        model=model,
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
        f.write(f"GPU优化批量提取 embedding 日志\n")
        f.write(f"总文件数: {len(wav_dict)}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {error_count}\n")
        f.write(f"批处理大小: {batch_size}\n\n")
        
        for result in results:
            if result["success"]:
                f.write(f"✓ {result['uttid']}: {result['embedding_path']}\n")
            else:
                f.write(f"✗ {result['uttid']}: {result['error']}\n")
    
    print(f"\n批量处理完成:")
    print(f"总文件数: {len(wav_dict)}")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"日志保存到: {log_path}")
    
    return results


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description="GPU优化的批量提取 Whisper embedding")
    parser.add_argument("wav_scp", help="wav.scp 文件路径")
    parser.add_argument("--output_dir", "-o", default="embeddings", help="输出目录")
    parser.add_argument("--model", default="base", help="Whisper 模型大小")
    parser.add_argument("--device", default="auto", help="设备类型 (auto/cpu/cuda)")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="批处理大小")
    parser.add_argument("--num_workers", "-w", type=int, default=4, help="工作进程数")
    parser.add_argument("--preserve_structure", action="store_true", default=True, help="保持原始目录结构")
    # parser.add_argument("--flat_output", action="store_true", default=False, help="使用扁平输出结构（所有文件在同一目录）")
    parser.add_argument("--single", help="处理单个音频文件（用于测试）")
    parser.add_argument("--dir_depth", type=int, default=3, help="保留原始目录深度")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 设置目录结构选项
    preserve_structure = args.preserve_structure
    dir_depth = args.dir_depth
    print(f"使用设备: {device}")
    print(f"加载模型: {args.model}")
    print(f"目录结构: {'层次化' if preserve_structure else '扁平化'}")
    print(f"目录深度: {dir_depth}")
    # 加载模型
    model = whisper.load_model(args.model, device=device)
    
    if args.single:
        # 处理单个文件（测试模式）
        print(f"测试模式: 处理单个文件 {args.single}")
        result = extract_embedding_single(
            model=model,
            audio_path=args.single,
            uttid="test",
            output_dir=args.output_dir,
            preserve_structure=preserve_structure
        )
        
        if result["success"]:
            print(f"✓ 成功处理: {result['uttid']}")
            print(f"  Embedding 形状: {result['embedding_shape']}")
            print(f"  保存路径: {result['embedding_path']}")
        else:
            print(f"✗ 处理失败: {result['error']}")
    else:
        # 批量处理
        if not os.path.exists(args.wav_scp):
            print(f"错误: wav.scp 文件不存在: {args.wav_scp}")
            sys.exit(1)
        
        print(f"开始GPU优化批量处理: {args.wav_scp}")
        results = batch_extract_embeddings_optimized(
            model=model,
            wav_scp_path=args.wav_scp,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            preserve_structure=preserve_structure,
            dir_depth=dir_depth
        )


if __name__ == "__main__":
    main()
