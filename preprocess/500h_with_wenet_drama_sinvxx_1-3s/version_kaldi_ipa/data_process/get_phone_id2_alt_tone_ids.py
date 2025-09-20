import re
from collections import defaultdict
from typing import Dict, List, Tuple
import sys
import json

# 类型别名（兼容 Python 3.8）
ToneGroups = Dict[str, Dict[int, int]]     # 例如: {"ing": {1: 101, 2: 102, 3: 103}}
Phoneme2ID = Dict[str, int]                # 例如: {"ing3": 103, "q": 3}
ID2Alts    = Dict[int, List[int]]          # 例如: {103: [101, 102, 104]}

_TONE_RE = re.compile(r"^([a-z]+)([1-5])$")  # 仅把结尾为 1..5 的 token 视作“带声调的韵母”

def build_tone_alt_dict(path: str) -> Tuple[ID2Alts, ToneGroups, Phoneme2ID]:
    """
    读取 'symbol id' 文本表，生成：
      - id2_alt_tone_ids: {带声调韵母的ID: [同一韵母其它声调的ID(按1..5排序)]}
      - final2_tone2id  : {韵母去声调: {tone: id}}
      - phoneme2id      : {原始符号: id}

    规则：
      1) 只把末尾带 1..5 的 token 当作“带声调的韵母”（如 ing3, uo2, er4）
      2) 其它（如声母 q, zh, ch，或无声调 ee/ii/uu 等）不参与韵母分组，但会记录到 phoneme2id
      3) 值列表不包含自身（“不同声调”）
    """
    phoneme2id: Phoneme2ID = {}
    final2_tone2id: ToneGroups = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            line = ln.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            sym = parts[0]
            # 允许行尾有注释/多列，取最后一列做 id
            try:
                pid = int(parts[-1])
            except ValueError:
                # 跳过无法解析 id 的行
                continue

            phoneme2id[sym] = pid

            m = _TONE_RE.fullmatch(sym)
            if m:
                base_final = m.group(1)         # 去掉声调的韵母，如 "ing"
                tone = int(m.group(2))          # 1..5
                final2_tone2id[base_final][tone] = pid

    # 为每个“带声调韵母ID”生成其它声调 ID 列表（按 tone 升序）
    id2_alt_tone_ids: ID2Alts = {}
    for base_final, tone2id in final2_tone2id.items():
        tones_sorted = sorted(tone2id.keys())  # 1..5 中实际存在的那些
        for tone in tones_sorted:
            this_id = tone2id[tone]
            # “不同声调” => 排除自身
            alt_ids = [tone2id[t] for t in tones_sorted if t != tone]
            id2_alt_tone_ids[this_id] = alt_ids

    return id2_alt_tone_ids, final2_tone2id, phoneme2id


# ==== 使用示例 ====
if __name__ == "__main__":
    phones_txt = sys.argv[1]   # 你的音素表路径，每行形如：ing3 4
    #phones_txt = "phones.txt"   # 你的音素表路径，每行形如：ing3 4
    out_alt_tone = sys.argv[2]
    id2_alt, final_groups, ph2id = build_tone_alt_dict(phones_txt)
    with open(out_alt_tone, 'w', encoding='utf-8') as f:
        json.dump(id2_alt, f, indent=2)

    ## 举例打印某个韵母族
    #target_final = "ing"
    #if target_final in final_groups:
    #    print(f"{target_final} 的 tone→id：", final_groups[target_final])
    #    for tone, pid in sorted(final_groups[target_final].items()):
    #        print(f"  id={pid} 的其它声调ID：", id2_alt.get(pid, []))
