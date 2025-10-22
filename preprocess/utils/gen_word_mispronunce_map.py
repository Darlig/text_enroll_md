#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast neighbor map for large lexicons (200k+) using pattern indexing.

Supports:
  --allow-sub / --allow-ins / --allow-del
  --max-distance D   # keep neighbors with Levenshtein distance <= D
  --cap-per-k        # optional cap per wildcard/deletion order to curb explosion

When D>=2, mixed operations (e.g., SUB+INS) are allowed as long as their op types
are enabled by the allow- switches.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Any, Set
import argparse, json, multiprocessing as mp
from itertools import combinations

SEP = " "

# ---------------- I/O ----------------
def load_lexicon(path: str) -> Dict[str, List[str]]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: list(v) for k, v in data.items()}
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

# ---------- pattern generators ----------
def wildcard_patterns_k(phones: List[str], k_max: int, cap_per_k: int|None=None) -> Iterable[str]:
    """Generate all patterns replacing EXACTLY k positions by '*' for k=1..k_max."""
    L = len(phones)
    for k in range(1, min(k_max, L)+1):
        c = 0
        for idxs in combinations(range(L), k):
            s = list(phones)
            for i in idxs:
                s[i] = "*"
            yield SEP.join(s)
            c += 1
            if cap_per_k is not None and c >= cap_per_k:
                break

def wildcard_patterns_exact_k(phones: List[str], k: int, cap_per_k: int|None=None) -> Iterable[str]:
    """Generate patterns with EXACTLY k wildcards. If k==0, yield the original sequence."""
    L = len(phones)
    if k == 0:
        yield SEP.join(phones)
        return
    c = 0
    for idxs in combinations(range(L), k):
        s = list(phones)
        for i in idxs:
            s[i] = "*"
        yield SEP.join(s)
        c += 1
        if cap_per_k is not None and c >= cap_per_k:
            break

def deletion_keys_k(phones: List[str], k_max: int, cap_per_k: int|None=None) -> Iterable[List[str]]:
    """Yield sequences (as list[str]) after deleting EXACTLY k positions, for k=1..k_max."""
    L = len(phones)
    for k in range(1, min(k_max, L)+1):
        c = 0
        for idxs in combinations(range(L), k):
            keep = [phones[i] for i in range(L) if i not in idxs]
            yield keep
            c += 1
            if cap_per_k is not None and c >= cap_per_k:
                break

# ---------- Levenshtein (<=D) with traceback ----------
def align_ops_leqD(p: List[str], q: List[str], D: int) -> Tuple[List[str] | None, int]:
    """Return (ops, dist) if dist <= D else (None, dist> D)."""
    lp, lq = len(p), len(q)
    if abs(lp - lq) > D:
        return None, D+1

    # banded DP (Ukkonen) distance
    INF = D + 1
    prev = [j if j <= D else INF for j in range(lq+1)]
    for i in range(1, lp+1):
        cur = [INF]*(lq+1)
        j_lo = max(1, i-D)
        j_hi = min(lq, i+D)
        cur[0] = i if i <= D else INF
        for j in range(j_lo, j_hi+1):
            sub  = prev[j-1] + (0 if p[i-1]==q[j-1] else 1)
            dele = prev[j] + 1
            ins  = cur[j-1] + 1
            cur[j] = min(sub, dele, ins)
        prev = cur
    dist = prev[lq]
    if dist > D:
        return None, dist

    # full DP for traceback
    dp = [[0]*(lq+1) for _ in range(lp+1)]
    for i in range(lp+1): dp[i][0] = i
    for j in range(lq+1): dp[0][j] = j
    for i in range(1, lp+1):
        for j in range(1, lq+1):
            dp[i][j] = min(
                dp[i-1][j-1] + (0 if p[i-1]==q[j-1] else 1),
                dp[i-1][j] + 1,
                dp[i][j-1] + 1
            )
    ops: List[str] = []
    i, j = lp, lq
    while i>0 or j>0:
        if i>0 and j>0 and dp[i][j] == dp[i-1][j-1] + (0 if p[i-1]==q[j-1] else 1):
            ops.append("M" if p[i-1]==q[j-1] else "SUB"); i-=1; j-=1
        elif i>0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append("DEL"); i-=1
        else:
            ops.append("INS"); j-=1
    ops.reverse()
    return ops, dist

# ---------- same-length neighbors via k-wildcards ----------
def same_length_neighbors(ws: List[str],
                          lex: Dict[str, List[str]],
                          Dmax: int,
                          allow_sub: bool, allow_ins: bool, allow_del: bool,
                          conf_pairs: Set[Tuple[str,str]]|None,
                          cap_per_k: int|None,
                          max_neighbors_per_word: int) -> Dict[str, List[Dict[str, Any]]]:
    out = {w: [] for w in ws}
    if not ws or Dmax <= 0:
        return out

    # wildcard buckets for k=1..Dmax
    pat2words: Dict[str, List[str]] = defaultdict(list)
    for w in ws:
        p = lex[w]
        for pat in wildcard_patterns_k(p, Dmax, cap_per_k):
            pat2words[pat].append(w)

    # pair candidates within each bucket
    seen_pairs = set()
    for group in pat2words.values():
        if len(group) < 2: continue
        G = group
        for i in range(len(G)):
            w1 = G[i]; p1 = lex[w1]
            for j in range(i+1, len(G)):
                w2 = G[j]; p2 = lex[w2]
                if (w1,w2) in seen_pairs: continue
                seen_pairs.add((w1,w2))
                ops, dist = align_ops_leqD(p1, p2, Dmax)
                if ops is None or not (0 < dist <= Dmax):
                    continue
                # op-type filter (allows mixing)
                if (not allow_sub and "SUB" in ops) or (not allow_ins and "INS" in ops) or (not allow_del and "DEL" in ops):
                    continue
                # optional confusion whitelist for pure single SUB
                if dist==1 and ops.count("SUB")==1 and ops.count("INS")==0 and ops.count("DEL")==0 and conf_pairs:
                    kpos = next(k for k,o in enumerate(ops) if o=="SUB")
                    if (p1[kpos], p2[kpos]) not in conf_pairs and (p2[kpos], p1[kpos]) not in conf_pairs:
                        continue
                if len(out[w1]) < max_neighbors_per_word:
                    out[w1].append(make_entry(w2, p2, ops))
                if len(out[w2]) < max_neighbors_per_word:
                    out[w2].append(make_entry(w1, p1, ops))
    for w in out:
        out[w].sort(key=lambda e: (e["distance"], e["neighbor"]))
    return out

# ---------- cross-length neighbors with delete+k-wildcards (supports INS+SUB etc.) ----------
def cross_length_neighbors(bucket_words_by_len: Dict[int, List[str]],
                           lex: Dict[str, List[str]],
                           Dmax: int,
                           allow_sub: bool, allow_ins: bool, allow_del: bool,
                           cap_per_k: int|None,
                           max_neighbors_per_word: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    For length difference k in 1..Dmax:
      - For each longer word, delete exactly k phones to get base seq Qk (many variants).
      - For r in 0..(Dmax-k), build wildcard patterns with exactly r '*' on Qk and index them.
      - For each shorter word P of length L, for the same r generate P's r-wildcards, probe the index.
      - Validate matches by DP (<=Dmax) and op-type filter. This recalls mixed ops like INS+SUB.
    """
    out = {w: [] for _, ws in bucket_words_by_len.items() for w in ws}
    if Dmax <= 0:
        return out

    lengths = sorted(bucket_words_by_len.keys())
    for L in lengths:
        short_ws = bucket_words_by_len[L]
        if not short_ws: continue

        for k in range(1, Dmax+1):
            long_ws = bucket_words_by_len.get(L+k, [])
            if not long_ws: continue

            # Build index: pattern -> list of (wlong, base_seq_after_del)
            pat2longs: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
            # For each longer word, enumerate all delete-k variants, then add r-wildcards on them (r=0..Dmax-k)
            for wlong in long_ws:
                q = lex[wlong]
                for base_seq in deletion_keys_k(q, k, cap_per_k):
                    R = Dmax - k
                    for r in range(0, R+1):
                        for pat in wildcard_patterns_exact_k(base_seq, r, cap_per_k):
                            pat2longs[pat].append((wlong, base_seq))

            # Probe with shorter words: for same r (0..R), generate r-wildcards and look up
            for wshort in short_ws:
                p = lex[wshort]
                R = Dmax - k
                for r in range(0, R+1):
                    for pat in wildcard_patterns_exact_k(p, r, cap_per_k):
                        cand = pat2longs.get(pat, [])
                        if not cand: continue
                        for wlong, base_seq in cand:
                            q = lex[wlong]
                            # validate both directions (short->long, long->short)
                            ops, dist = align_ops_leqD(p, q, Dmax)
                            if ops is not None and 0 < dist <= Dmax:
                                if not ((not allow_sub and "SUB" in ops) or (not allow_ins and "INS" in ops) or (not allow_del and "DEL" in ops)):
                                    if len(out[wshort]) < max_neighbors_per_word:
                                        out[wshort].append(make_entry(wlong, q, ops))
                            opsr, distr = align_ops_leqD(q, p, Dmax)
                            if opsr is not None and 0 < distr <= Dmax:
                                if not ((not allow_sub and "SUB" in opsr) or (not allow_ins and "INS" in opsr) or (not allow_del and "DEL" in opsr)):
                                    if len(out[wlong]) < max_neighbors_per_word:
                                        out[wlong].append(make_entry(wshort, p, opsr))

    for w in out:
        out[w].sort(key=lambda e: (e["distance"], e["neighbor"]))
    return out

# ---------- Orchestration ----------
def chunk_by_length(lex: Dict[str, List[str]]) -> Dict[int, List[str]]:
    buckets = defaultdict(list)
    for w, p in lex.items():
        buckets[len(p)].append(w)
    return buckets

def run_fast_neighbors(lex: Dict[str, List[str]],
                       allow_sub: bool,
                       allow_ins: bool,
                       allow_del: bool,
                       max_distance: int,
                       conf_pairs: Set[Tuple[str,str]]|None,
                       max_neighbors_per_word: int,
                       cap_per_k: int|None,
                       workers: int = max(1, mp.cpu_count()//2)) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {w: {"canonical_pron": lex[w], "neighbors": []} for w in lex}
    buckets = chunk_by_length(lex)
    Dmax = max(1, int(max_distance))

    # same-length (parallel per length)
    args = []
    for _, ws in buckets.items():
        args.append((ws, lex, Dmax, allow_sub, allow_ins, allow_del, conf_pairs, cap_per_k, max_neighbors_per_word))
    with mp.Pool(processes=workers) as pool:
        parts = pool.starmap(same_length_neighbors, args)
    for part in parts:
        for w, entries in part.items():
            result[w]["neighbors"].extend(entries)

    # cross-length (only if INS/DEL allowed)
    if allow_ins or allow_del:
        cross_map = cross_length_neighbors(buckets, lex, Dmax, allow_sub, allow_ins, allow_del,
                                           cap_per_k, max_neighbors_per_word)
        for w, entries in cross_map.items():
            result[w]["neighbors"].extend(entries)

    for w in result:
        result[w]["neighbors"].sort(key=lambda e: (e["distance"], e["neighbor"]))
    return result

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--out", required=True)

    # per-op switches (mutually exclusive pairs)
    group_sub = ap.add_mutually_exclusive_group()
    group_sub.add_argument("--allow-sub",    dest="allow_sub", action="store_true",  help="Allow SUB.")
    group_sub.add_argument("--no-allow-sub", dest="allow_sub", action="store_false", help="Disallow SUB.")
    ap.set_defaults(allow_sub=True)

    group_ins = ap.add_mutually_exclusive_group()
    group_ins.add_argument("--allow-ins",    dest="allow_ins", action="store_true",  help="Allow INS.")
    group_ins.add_argument("--no-allow-ins", dest="allow_ins", action="store_false", help="Disallow INS.")
    ap.set_defaults(allow_ins=False)

    group_del = ap.add_mutually_exclusive_group()
    group_del.add_argument("--allow-del",    dest="allow_del", action="store_true",  help="Allow DEL.")
    group_del.add_argument("--no-allow-del", dest="allow_del", action="store_false", help="Disallow DEL.")
    ap.set_defaults(allow_del=False)

    # maximum distance (keep pairs with dist <= D)
    ap.add_argument("--max-distance", type=int, default=1,
                    help="Maximum Levenshtein distance to keep (e.g., 1 or 2). Default: 1")

    # extras
    ap.add_argument("--confusions", default="", help="Optional whitelist file: 'a b' per line for pure single SUB (dist=1).")
    ap.add_argument("--max-neighbors-per-word", type=int, default=50)
    ap.add_argument("--cap-per-k", type=int, default=None,
                    help="Cap patterns per order (wildcards/deletions) to avoid combinatorial explosion. Default: unlimited.")
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
            # confusions only used in same-length pure SUB=1 scenario
                if not ln or ln.startswith("#"): continue
                a, b = ln.split()
                conf_pairs.add((a, b))

    neighbors = run_fast_neighbors(
        lex=lex,
        allow_sub=args.allow_sub,
        allow_ins=args.allow_ins,
        allow_del=args.allow_del,
        max_distance=args.max_distance,
        conf_pairs=conf_pairs,
        max_neighbors_per_word=args.max_neighbors_per_word,
        cap_per_k=args.cap_per_k,
        workers=args.workers
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(neighbors, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {args.out}. words={len(lex)}  ops: sub={args.allow_sub}, ins={args.allow_ins}, del={args.allow_del}  maxD={args.max_distance}")

if __name__ == "__main__":
    main()