#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast neighbor map for large lexicons (200k+) using pattern indexing.

Now supports per-op switches:
  --allow-sub (default: True)
  --allow-ins (default: False)
  --allow-del (default: False)

Output schema:
{
  "<word>": {
    "canonical_pron": ["..."],
    "neighbors": [
      {
        "neighbor": "<neighbor word>",
        "candidate_pron": ["..."],
        "distance": 1,
        "counts": { "sub": 1, "ins": 0, "del": 0 },
        "ops": ["SUB","M","M","M"]
      },
      ...
    ]
  },
  ...
}

Two construction paths:
- SUB-only, distance=1 (equal length) via wildcard index -> O(N·L)
- INS/DEL (ED<=1) via deletion index (SymSpell-like) -> O(N·L)
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Any, Set
import argparse, json, os, sys, multiprocessing as mp

GAP = "∅"

# ---------------- I/O ----------------
def load_lexicon(path: str) -> Dict[str, List[str]]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: list(v) for k, v in data.items()}
    # TSV/space-delimited: word p1 p2 ...
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            parts = ln.split()
            if len(parts) < 2: continue
            out[parts[0]] = parts[1:]
    return out

# ------------- helpers (output packing) -------------
def count_ops_from_ops_list(ops_list: List[str]) -> Dict[str, int]:
    c = {"sub": 0, "ins": 0, "del": 0}
    for op in ops_list:
        if op == "SUB": c["sub"] += 1
        elif op == "INS": c["ins"] += 1
        elif op == "DEL": c["del"] += 1
    return c

def make_entry(neighbor_word: str,
               candidate_pron: List[str],
               ops_strings: List[str]) -> Dict[str, Any]:
    distance = sum(1 for op in ops_strings if op in ("SUB","INS","DEL"))
    return {
        "neighbor": neighbor_word,
        "candidate_pron": candidate_pron,
        "distance": distance,
        "counts": count_ops_from_ops_list(ops_strings),
        "ops": ops_strings
    }

# ------------- Mode A: SUB-only (equal length, dist==1) -------------
def wildcard_patterns(phones: List[str], sep: str=" ") -> Iterable[str]:
    for i in range(len(phones)):
        yield sep.join(phones[:i] + ["*"] + phones[i+1:])

def build_wildcard_index(bucket_words: List[str],
                         lex: Dict[str, List[str]],
                         sep: str=" ") -> Dict[str, List[str]]:
    idx = defaultdict(list)
    for w in bucket_words:
        p = lex[w]
        for pat in wildcard_patterns(p, sep):
            idx[pat].append(w)
    return idx

def subonly_neighbors_for_bucket(bucket_words: List[str],
                                 lex: Dict[str, List[str]],
                                 allow_sub: bool,
                                 conf_pairs: Set[Tuple[str,str]]|None = None,
                                 max_neighbors_per_word: int = 50) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns {word: [NeighborEntry,...]} for words with same length.
    Each entry: neighbor, candidate_pron, distance, counts, ops
    """
    result: Dict[str, List[Dict[str, Any]]] = {w: [] for w in bucket_words}
    if not bucket_words or not allow_sub:
        return result

    L = len(lex[bucket_words[0]])
    idx = build_wildcard_index(bucket_words, lex)

    for _, words in idx.items():
        if len(words) < 2:
            continue
        for i, w1 in enumerate(words):
            p1 = lex[w1]
            for w2 in words[i+1:]:
                p2 = lex[w2]
                # differ at exactly one position
                diff_pos = -1
                diff_cnt = 0
                for k in range(L):
                    if p1[k] != p2[k]:
                        diff_cnt += 1
                        diff_pos = k
                        if diff_cnt > 1: break
                if diff_cnt != 1:
                    continue
                # whitelist (optional)
                if conf_pairs is not None:
                    pair = (p1[diff_pos], p2[diff_pos])
                    rev  = (p2[diff_pos], p1[diff_pos])
                    if (pair not in conf_pairs) and (rev not in conf_pairs):
                        continue

                ops12 = ["SUB" if k==diff_pos else "M" for k in range(L)]
                ops21 = ["SUB" if k==diff_pos else "M" for k in range(L)]

                if len(result[w1]) < max_neighbors_per_word:
                    result[w1].append(make_entry(w2, p2, ops12))
                if len(result[w2]) < max_neighbors_per_word:
                    result[w2].append(make_entry(w1, p1, ops21))

    for w in result:
        result[w].sort(key=lambda e: (e["distance"], e["neighbor"]))
    return result

# ------------- Mode B: INS/DEL via deletion index (ED<=1) -------------
def deletion_keys(phones: List[str], sep: str=" ") -> Iterable[str]:
    for i in range(len(phones)):
        yield sep.join(phones[:i] + phones[i+1:])

def build_deletion_index(bucket_words: List[str],
                         lex: Dict[str, List[str]],
                         sep: str=" ") -> Dict[str, List[str]]:
    idx = defaultdict(list)
    for w in bucket_words:
        for key in deletion_keys(lex[w], sep):
            idx[key].append(w)
    return idx

def align_ed1_ops(p: List[str], q: List[str]) -> List[str] | None:
    """
    Edit distance <= 1:
      - equal length: exactly 1 SUB -> ops = ["M"/"SUB"] length L
      - len diff +1: exactly 1 INS  -> ops includes one "INS"
      - len diff -1: exactly 1 DEL  -> ops includes one "DEL"
    Returns ops list of strings or None if not ED<=1.
    NOTE: This function returns only the ops sequence (not aligned ref/hyp phones),
          which is sufficient for counts/distance and schema here.
    """
    # equal length: exactly 1 SUB
    if len(p) == len(q):
        diff = [i for i,(a,b) in enumerate(zip(p,q)) if a!=b]
        if len(diff) != 1:
            return None
        k = diff[0]
        return ["SUB" if i==k else "M" for i in range(len(p))]

    # insertion wrt p (q longer by 1)
    if len(p) + 1 == len(q):
        ops: List[str] = []
        i=j=0
        ins_done = False
        while i < len(p) and j < len(q):
            if p[i] == q[j]:
                ops.append("M")
                i += 1; j += 1
            else:
                if ins_done:
                    return None
                ops.append("INS")
                j += 1
                ins_done = True
        # handle tail
        if j < len(q):
            # remaining in q must be exactly one and INS not done yet
            if ins_done or (j != len(q)-1):
                return None
            ops.append("INS")
            j += 1
            ins_done = True
        # if p has tail left, invalid for INS case
        if i < len(p):
            return None
        return ops

    # deletion wrt p (q shorter by 1)
    if len(p) == len(q) + 1:
        ops: List[str] = []
        i=j=0
        del_done = False
        while i < len(p) and j < len(q):
            if p[i] == q[j]:
                ops.append("M")
                i += 1; j += 1
            else:
                if del_done:
                    return None
                ops.append("DEL")
                i += 1
                del_done = True
        # handle tail in p
        if i < len(p):
            if del_done or (i != len(p)-1):
                return None
            ops.append("DEL")
            i += 1
            del_done = True
        # if q has tail left, invalid for DEL case
        if j < len(q):
            return None
        return ops

    return None

def ed1_neighbors_for_bucket(bucket_words_by_len: Dict[int, List[str]],
                             lex: Dict[str, List[str]],
                             allow_sub: bool,
                             allow_ins: bool,
                             allow_del: bool,
                             conf_pairs: Set[Tuple[str,str]]|None = None,
                             max_neighbors_per_word: int = 50) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build ED<=1 neighbors across length L and L±1 using deletion index.
    Returns {word: [NeighborEntry,...]} obeying per-op switches.
    """
    out = {w: [] for _, ws in bucket_words_by_len.items() for w in ws}

    # 1) same-length SUB-only via wildcard (obey allow_sub)
    for _, ws in bucket_words_by_len.items():
        sub_map = subonly_neighbors_for_bucket(ws, lex, allow_sub, conf_pairs, max_neighbors_per_word)
        for w, lst in sub_map.items():
            out[w].extend(lst)

    # 2) INS/DEL between L and L+1 via deletion index (only index the longer side!)
    for L in bucket_words_by_len:
        ws_short = bucket_words_by_len[L]
        ws_long  = bucket_words_by_len.get(L+1, [])
        if not ws_short or not ws_long:
            continue

        # build deletion index ONLY for the longer words:
        # key = long_word with one phone removed  ==> maps to that long word
        long_del_idx: Dict[str, List[str]] = defaultdict(list)
        for w_long in ws_long:
            q = lex[w_long]
            for i in range(len(q)):
                key = " ".join(q[:i] + q[i+1:])
                long_del_idx[key].append(w_long)

        # now, for each short word, look up its FULL sequence in the long_del_idx
        for w_short in ws_short:
            p = lex[w_short]
            key = " ".join(p)
            cand_longs = long_del_idx.get(key, [])
            if not cand_longs:
                continue

            for w_long in cand_longs:
                q = lex[w_long]
                # forward (short -> long): should be INS wrt short
                ops = align_ed1_ops(p, q)
                if ops is not None:
                    has_sub = any(op == "SUB" for op in ops)
                    has_ins = any(op == "INS" for op in ops)
                    has_del = any(op == "DEL" for op in ops)
                    if not ((has_sub and not allow_sub) or (has_ins and not allow_ins) or (has_del and not allow_del)):
                        if len(out[w_short]) < max_neighbors_per_word:
                            out[w_short].append(make_entry(w_long, q, ops))

                # reverse (long -> short): should be DEL wrt long
                ops_rev = align_ed1_ops(q, p)
                if ops_rev is not None:
                    has_sub_r = any(op == "SUB" for op in ops_rev)
                    has_ins_r = any(op == "INS" for op in ops_rev)
                    has_del_r = any(op == "DEL" for op in ops_rev)
                    if not ((has_sub_r and not allow_sub) or (has_ins_r and not allow_ins) or (has_del_r and not allow_del)):
                        if len(out[w_long]) < max_neighbors_per_word:
                            out[w_long].append(make_entry(w_short, p, ops_rev))

    # # 2) INS/DEL between L and L+1 via deletion index (obey allow_ins/allow_del)
    # # If both disabled, skip
    # if not allow_ins and not allow_del:
    #     # still return what SUB-only added
    #     for w in out:
    #         out[w].sort(key=lambda e: (e["distance"], e["neighbor"]))
    #     return out

    # for L in bucket_words_by_len:
    #     wsL   = bucket_words_by_len[L]
    #     wsLp1 = bucket_words_by_len.get(L+1, [])
    #     if not wsL or not wsLp1:
    #         continue
    #     idx_L  = build_deletion_index(wsL,  lex)
    #     idx_L1 = build_deletion_index(wsLp1, lex)

    #     # words of len L can INS to len L+1 via shared delete-key
    #     for key, wlist_long in idx_L1.items():
    #         short_list = idx_L.get(key, [])
    #         if not short_list:
    #             continue
    #         for w_short in short_list:
    #             p = lex[w_short]
    #             for w_long in wlist_long:
    #                 q = lex[w_long]
    #                 ops = align_ed1_ops(p, q)
    #                 if ops is None:
    #                     continue
    #                 # filter by allowed ops
    #                 has_sub = any(op == "SUB" for op in ops)
    #                 has_ins = any(op == "INS" for op in ops)
    #                 has_del = any(op == "DEL" for op in ops)
    #                 if (has_sub and not allow_sub) or (has_ins and not allow_ins) or (has_del and not allow_del):
    #                     continue
    #                 if len(out[w_short]) < max_neighbors_per_word:
    #                     out[w_short].append(make_entry(w_long, q, ops))

    #                 # reverse direction
    #                 ops_rev = align_ed1_ops(q, p)
    #                 if ops_rev is not None:
    #                     has_sub_r = any(op == "SUB" for op in ops_rev)
    #                     has_ins_r = any(op == "INS" for op in ops_rev)
    #                     has_del_r = any(op == "DEL" for op in ops_rev)
    #                     if (has_sub_r and not allow_sub) or (has_ins_r and not allow_ins) or (has_del_r and not allow_del):
    #                         pass
    #                     elif len(out[w_long]) < max_neighbors_per_word:
    #                         out[w_long].append(make_entry(w_short, p, ops_rev))

    for w in out:
        out[w].sort(key=lambda e: (e["distance"], e["neighbor"]))
    return out

# ------------- Orchestration & CLI -------------
def chunk_by_length(lex: Dict[str, List[str]]) -> Dict[int, List[str]]:
    buckets = defaultdict(list)
    for w, p in lex.items():
        buckets[len(p)].append(w)
    return buckets

def run_fast_neighbors(lex: Dict[str, List[str]],
                       allow_sub: bool,
                       allow_ins: bool,
                       allow_del: bool,
                       conf_pairs: Set[Tuple[str,str]]|None,
                       max_neighbors_per_word: int,
                       workers: int = max(1, mp.cpu_count()//2)) -> Dict[str, Dict[str, Any]]:
    """
    Return:
      {
        word: {
          "canonical_pron": [...],
          "neighbors": [NeighborEntry,...]
        },
        ...
      }
    """
    result: Dict[str, Dict[str, Any]] = {w: {"canonical_pron": lex[w], "neighbors": []} for w in lex}

    # Fast path: if only SUB is allowed, we can parallelize purely by length buckets
    only_sub = allow_sub and not allow_ins and not allow_del

    buckets = chunk_by_length(lex)
    print(f"allow_sub: {allow_sub}, allow_ins: {allow_ins}, allow_del: {allow_del}")

    if only_sub:
        print("Only SUB is allowed")
        args = [(ws, lex, allow_sub, conf_pairs, max_neighbors_per_word) for _, ws in buckets.items()]
        with mp.Pool(processes=workers) as pool:
            parts = pool.starmap(subonly_neighbors_for_bucket, args)
        for part in parts:
            for w, entries in part.items():
                result[w]["neighbors"].extend(entries)
        return result

    print("INS/DEL is allowed")
    # Otherwise we need ED<=1 path including potential INS/DEL
    neighbor_map = ed1_neighbors_for_bucket(buckets, lex, allow_sub, allow_ins, allow_del,
                                            conf_pairs, max_neighbors_per_word)
    for w, entries in neighbor_map.items():
        result[w]["neighbors"].extend(entries)
    return result

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--out", required=True)
    # per-op switches
    ap.add_argument("--allow-sub", action="store_true", default=False, help="Allow SUB (equal length, dist=1). Default: True.")
    ap.add_argument("--allow-ins", action="store_true", default=False, help="Allow INS (length +1). Default: False.")
    ap.add_argument("--allow-del", action="store_true", default=False, help="Allow DEL (length -1). Default: False.")
    # extras
    ap.add_argument("--confusions", default="", help="Optional confusion whitelist file: 'a b' per line for allowed (a,b). Only affects SUB.")
    ap.add_argument("--max-neighbors-per-word", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()//2))
    return ap.parse_args()

def main():
    args = parse_args()
    lex = load_lexicon(args.lexicon)

    conf_pairs: Set[Tuple[str,str]] | None = None
    if args.confusions:
        conf_pairs = set()
        with open(args.confusions, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"): continue
                a, b = ln.split()
                conf_pairs.add((a, b))

    neighbors = run_fast_neighbors(
        lex=lex,
        allow_sub=args.allow_sub,
        allow_ins=args.allow_ins,
        allow_del=args.allow_del,
        conf_pairs=conf_pairs,
        max_neighbors_per_word=args.max_neighbors_per_word,
        workers=args.workers
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(neighbors, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {args.out}. words={len(lex)}  ops: sub={args.allow_sub}, ins={args.allow_ins}, del={args.allow_del}")

if __name__ == "__main__":
    main()