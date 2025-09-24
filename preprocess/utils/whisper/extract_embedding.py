"""
最简化的 Whisper 转录实现
只使用三个核心函数：log_mel_spectrogram, pad_or_trim, decode
"""

import torch
import whisper
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm
from whisper.decoding import DecodingOptions
from whisper.audio import N_FRAMES


def extract_embedding(model, audio_path, uttid=None, save_embedding=True, output_dir="embeddings"):
    """
    提取单个音频文件的 embedding
    
    Parameters
    ----------
    model : Whisper
        Whisper 模型实例
    audio_path : str
        音频文件路径
    uttid : str
        音频ID，如果为None则从文件名提取
    save_embedding : bool
        是否保存 embedding
    output_dir : str
        embedding 保存目录
        
    Returns
    -------
    dict
        包含 embedding 路径和形状的字典
    """
    try:
        # 1. 加载音频
        audio_data = whisper.load_audio(audio_path)
        
        # 2. 生成梅尔频谱图
        mel = whisper.log_mel_spectrogram(audio_data, model.dims.n_mels)
        
        # 3. 填充到固定长度
        mel = whisper.pad_or_trim(mel, N_FRAMES)
        
        # 4. 移动到设备并添加批次维度
        mel = mel.to(model.device)
        if mel.dim() == 2:  # 如果是2D，添加批次维度
            mel = mel.unsqueeze(0)
        
        # 5. 获取音频编码器特征
        with torch.no_grad():
            audio_features = model.encoder(mel)
        
        # 6. 保存 embedding
        embedding_path = None
        if save_embedding:
            os.makedirs(output_dir, exist_ok=True)
            if uttid is None:
                uttid = os.path.basename(audio_path).replace('.wav', '')
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
    """
    加载 wav.scp 文件
    
    Parameters
    ----------
    wav_scp_path : str
        wav.scp 文件路径
        
    Returns
    -------
    dict
        音频ID到音频路径的映射
    """
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


def batch_extract_embeddings(model, wav_scp_path, output_dir="embeddings", device=None):
    """
    批量提取 embedding
    
    Parameters
    ----------
    model : Whisper
        Whisper 模型实例
    wav_scp_path : str
        wav.scp 文件路径
    output_dir : str
        输出目录
    device : str
        设备类型
        
    Returns
    -------
    list
        处理结果列表
    """
    # 加载 wav.scp 文件
    wav_dict = load_wav_scp(wav_scp_path)
    print(f"加载了 {len(wav_dict)} 个音频文件")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    success_count = 0
    error_count = 0
    
    # 批量处理
    for uttid, audio_path in tqdm(wav_dict.items(), desc="提取 embedding"):
        result = extract_embedding(
            model=model,
            audio_path=audio_path,
            uttid=uttid,
            save_embedding=True,
            output_dir=output_dir
        )
        
        results.append(result)
        
        if result["success"]:
            success_count += 1
        else:
            error_count += 1
            print(f"处理失败 {uttid}: {result['error']}")
    
    # 保存处理日志
    log_path = os.path.join(output_dir, "extraction_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"批量提取 embedding 日志\n")
        f.write(f"总文件数: {len(wav_dict)}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {error_count}\n\n")
        
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
    parser = argparse.ArgumentParser(description="批量提取 Whisper embedding")
    parser.add_argument("wav_scp", help="wav.scp 文件路径")
    parser.add_argument("--output_dir", "-o", default="embeddings", help="输出目录")
    parser.add_argument("--model", default="base", help="Whisper 模型大小")
    parser.add_argument("--device", default="auto", help="设备类型 (auto/cpu/cuda)")
    parser.add_argument("--single", help="处理单个音频文件（用于测试）")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    print(f"加载模型: {args.model}")
    
    # 加载模型
    model = whisper.load_model(args.model, device=device)
    
    if args.single:
        # 处理单个文件（测试模式）
        print(f"测试模式: 处理单个文件 {args.single}")
        result = extract_embedding(
            model=model,
            audio_path=args.single,
            save_embedding=True,
            output_dir=args.output_dir
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
        
        print(f"开始批量处理: {args.wav_scp}")
        results = batch_extract_embeddings(
            model=model,
            wav_scp_path=args.wav_scp,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
