# Whisper Embedding 批量提取工具

这个工具可以从 `wav.scp` 格式的音频索引文件中批量提取 Whisper embedding。

## 功能特性

- 支持 `wav.scp` 格式的音频索引文件
- 批量提取 Whisper 编码器特征
- 自动错误处理和日志记录
- 支持进度条显示
- 支持单文件测试模式

## 使用方法

### 1. 准备 wav.scp 文件

`wav.scp` 文件格式：
```
uttid1 /path/to/audio1.wav
uttid2 /path/to/audio2.wav
uttid3 /path/to/audio3.wav
```

### 2. 批量提取 embedding

```bash
# 基本用法
python preprocess/utils/whisper/extract_embedding.py your_wav.scp

# 指定输出目录
python preprocess/utils/whisper/extract_embedding.py your_wav.scp --output_dir embeddings

# 指定模型大小
python preprocess/utils/whisper/extract_embedding.py your_wav.scp --model large

# 指定设备
python preprocess/utils/whisper/extract_embedding.py your_wav.scp --device cuda
```

### 3. 单文件测试

```bash
# 测试单个音频文件
python preprocess/utils/whisper/extract_embedding.py dummy.scp --single /path/to/audio.wav
```

## 参数说明

- `wav_scp`: wav.scp 文件路径（必需）
- `--output_dir, -o`: 输出目录（默认: embeddings）
- `--model`: Whisper 模型大小（默认: base）
- `--device`: 设备类型 auto/cpu/cuda（默认: auto）
- `--single`: 处理单个音频文件（用于测试）

## 输出文件

### 1. Embedding 文件
- 文件名格式：`{uttid}.npy`
- 文件内容：numpy 数组，形状为 `(1, 1500, 512)`
- 数据类型：float32

### 2. 处理日志
- 文件名：`extraction_log.txt`
- 内容：处理统计信息和详细日志

## 示例

### 创建 wav.scp 文件
```bash
echo "ID0485W0189 preprocess/utils/ID0485W0189.wav" > example_wav.scp
echo "test_audio1 preprocess/utils/ID0485W0189.wav" >> example_wav.scp
```

### 批量处理
```bash
python preprocess/utils/whisper/extract_embedding.py example_wav.scp --output_dir my_embeddings
```

### 输出结果
```
使用设备: cuda
加载模型: base
开始批量处理: example_wav.scp
加载了 2 个音频文件
提取 embedding: 100%|██████████| 2/2 [00:00<00:00, 6.57it/s]

批量处理完成:
总文件数: 2
成功: 2
失败: 0
日志保存到: my_embeddings/extraction_log.txt
```

## 文件结构

```
my_embeddings/
├── ID0485W0189.npy          # embedding 文件
├── test_audio1.npy          # embedding 文件
└── extraction_log.txt       # 处理日志
```

## 加载 embedding

```python
import numpy as np

# 加载 embedding
embedding = np.load("my_embeddings/ID0485W0189.npy")
print(f"Embedding 形状: {embedding.shape}")  # (1, 1500, 512)
print(f"数据类型: {embedding.dtype}")        # float32
```

## 注意事项

1. 确保音频文件路径正确且可访问
2. 大模型（large）需要更多内存和计算时间
3. 建议使用 GPU 加速处理
4. 处理大量文件时注意磁盘空间
