#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast neighbor map for large lexicons (200k+) using pattern indexing.

Change magnitude is controlled by a ratio threshold:
    distance / length <= max_ratio
where "length" is the phone-seq length of the direction's source word.

New constraint:
    Per direction, INS + DEL <= 1
    (i.e., at most one insertion OR one deletion, not both, and not more than one.)
This implies |len(p) - len(q)| <= 1 for valid neighbors.

Supports:
  --allow-sub / --allow-ins / --allow-del
  --max-ratio R     # ratio in [0, 1], e.g., 0.2 means up to floor(0.2*len(src))
  --cap-per-k       # optional cap per wildcard/deletion order to curb explosion

When mixed ops are enabled, any combination (SUB/INS/DEL) is allowed
subject to the ratio threshold and INS+DEL<=1 constraint.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Any, Set
import argparse
import json
import multiprocessing as mp
import sys
from itertools import combinations
import math

# Constants
SEP = " "
OP_SUB = "SUB"
OP_INS = "INS"
OP_DEL = "DEL"
OP_MATCH = "M"

# ---------------- I/O ----------------
def load_lexicon(path: str) -> Dict[str, List[str]]:
    """Load lexicon from file (JSON or text format)."""
    try:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: list(v) for k, v in data.items()}

        out = {}
        with open(path, "r", encoding="utf-8") as f:
            for line_num, ln in enumerate(f, 1):
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) < 2:
                    print(f"Warning: Line {line_num} has insufficient parts, skipping", file=sys.stderr)
                    continue
                out[parts[0]] = parts[1:]
        return out
    except FileNotFoundError:
        print(f"Error: Lexicon file '{path}' not found", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{path}': {e}", file=sys.stderr)
        raise

# ------------- helpers (output packing) -------------
def count_ops_from_ops_list(ops_list: List[str]) -> Dict[str, int]:
    c = {"sub": 0, "ins": 0, "del": 0}
    for op in ops_list:
        if op == OP_SUB:
            c["sub"] += 1
        elif op == OP_INS:
            c["ins"] += 1
        elif op == OP_DEL:
            c["del"] += 1
    return c

def make_entry(neighbor_word: str,
               candidate_pron: List[str],
               ops_strings: List[str],
               src_len: int) -> Dict[str, Any]:
    """Create a neighbor entry with distance, ratio, and operation details.
    
    ratio = distance / src_len  (src_len = length of the source word in THIS direction)
    """
    distance = sum(1 for op in ops_strings if op in (OP_SUB, OP_INS, OP_DEL))
    ratio = distance / src_len if src_len > 0 else 0.0
    return {
        "neighbor": neighbor_word,
        "candidate_pron": candidate_pron,
        "distance": distance,
        "ratio": ratio,
        "counts": count_ops_from_ops_list(ops_strings),
        "ops": ops_strings
    }

def ops_allowed(ops: List[str], allow_sub: bool, allow_ins: bool, allow_del: bool) -> bool:
    """All ops must be allowed; enforce INS+DEL <= 1 per-direction."""
    ins_cnt = 0
    del_cnt = 0
    for op in ops:
        if op == OP_SUB:
            if not allow_sub:
                return False
        elif op == OP_INS:
            if not allow_ins:
                return False
            ins_cnt += 1
            if ins_cnt + del_cnt > 1:
                return False
        elif op == OP_DEL:
            if not allow_del:
                return False
            del_cnt += 1
            if ins_cnt + del_cnt > 1:
                return False
        # OP_MATCH has no effect
    return True

# ---------- pattern generators ----------
def wildcard_patterns_k(phones: List[str], k_max: int, cap_per_k: int|None=None) -> Iterable[str]:
    """Generate patterns replacing EXACTLY k positions by '*' for k=1..k_max."""
    L = len(phones)
    if k_max <= 0:
        return
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
    if k < 0:
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
    if k_max <= 0:
        return
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
    """Return (ops, dist) if dist<=D else (None, dist)."""
    lp, lq = len(p), len(q)
    if D < 0:
        return None, abs(lp - lq) + 1
    if abs(lp - lq) > D:
        return None, abs(lp - lq)

    dp = [[0] * (lq + 1) for _ in range(lp + 1)]
    for i in range(lp + 1): dp[i][0] = i
    for j in range(lq + 1): dp[0][j] = j

    for i in range(1, lp + 1):
        for j in range(1, lq + 1):
            match_cost = 0 if p[i-1] == q[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j-1] + match_cost,
                dp[i-1][j] + 1,
                dp[i][j-1] + 1
            )
    dist = dp[lp][lq]
    if dist > D:
        return None, dist

    ops: List[str] = []
    i, j = lp, lq
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_cost = 0 if p[i-1] == q[j-1] else 1
            if dp[i][j] == dp[i-1][j-1] + match_cost:
                ops.append(OP_MATCH if p[i-1] == q[j-1] else OP_SUB)
                i -= 1; j -= 1; continue
        if i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(OP_DEL); i -= 1
        else:
            ops.append(OP_INS); j -= 1
    ops.reverse()
    return ops, dist

# ---------- same-length neighbors via per-word kmax (ratio) ----------
def same_length_neighbors(ws: List[str],
                          lex: Dict[str, List[str]],
                          max_ratio: float,
                          allow_sub: bool, allow_ins: bool, allow_del: bool,
                          conf_pairs: Set[Tuple[str,str]] | None,
                          cap_per_k: int | None,
                          max_neighbors_per_word: int) -> Dict[str, List[Dict[str, Any]]]:
    """Same-length candidates built by wildcard patterns up to kmax=floor(ratio*len(word))."""
    out = {w: [] for w in ws}
    if not ws or max_ratio < 0:
        return out

    pat2words: Dict[str, List[str]] = defaultdict(list)
    for w in ws:
        p = lex[w]
        kmax = math.floor(max_ratio * len(p))
        for pat in wildcard_patterns_k(p, kmax, cap_per_k):
            pat2words[pat].append(w)

    seen_pairs = set()
    for group in pat2words.values():
        if len(group) < 2: continue
        for i in range(len(group)):
            w1 = group[i]; p1 = lex[w1]
            for j in range(i + 1, len(group)):
                w2 = group[j]; p2 = lex[w2]
                if (w1, w2) in seen_pairs: continue
                seen_pairs.add((w1, w2))

                # Same length → Dpair is the same either way
                Dpair = math.floor(max_ratio * len(p1))
                if Dpair <= 0:
                    continue

                ops, dist = align_ops_leqD(p1, p2, Dpair)
                if ops is None or dist <= 0:
                    continue
                # Enforce op-type switches + INS+DEL<=1 (implies same-length pairs are pure SUB)
                if not ops_allowed(ops, allow_sub, allow_ins, allow_del):
                    continue

                # Optional confusion whitelist for pure SUB=1
                if (dist == 1 and ops.count(OP_SUB) == 1 and 
                    ops.count(OP_INS) == 0 and ops.count(OP_DEL) == 0 and conf_pairs):
                    kpos = next(k for k, o in enumerate(ops) if o == OP_SUB)
                    if ((p1[kpos], p2[kpos]) not in conf_pairs and 
                        (p2[kpos], p1[kpos]) not in conf_pairs):
                        continue

                if len(out[w1]) < max_neighbors_per_word:
                    out[w1].append(make_entry(w2, p2, ops, src_len=len(p1)))
                if len(out[w2]) < max_neighbors_per_word:
                    out[w2].append(make_entry(w1, p1, ops, src_len=len(p2)))

    for w in out:
        out[w].sort(key=lambda e: (e["distance"], e["neighbor"]))
    return out

# ---------- cross-length neighbors with ratio (|len diff| = 1 only) ----------
def cross_length_neighbors(bucket_words_by_len: Dict[int, List[str]],
                           lex: Dict[str, List[str]],
                           max_ratio: float,
                           allow_sub: bool, allow_ins: bool, allow_del: bool,
                           cap_per_k: int | None,
                           max_neighbors_per_word: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Anchor on the SHORT side; only consider length difference k=1,
    because INS+DEL<=1 implies |len(p)-len(q)|<=1.

      For each short length L:
        kmax_short = floor(max_ratio * L)
        If kmax_short >= 1:
          consider long length L+1 (if exists)
          deletions on long side = 1
          allow extra wildcards r in 0..(kmax_short - 1)

    This pass captures pairs where the SHORT side also satisfies the ratio.
    If you also want pairs valid only from the LONG side perspective,
    run a symmetric pass anchoring on the LONG side (not needed in most pipelines).
    """
    out = {w: [] for _, ws in bucket_words_by_len.items() for w in ws}
    if max_ratio < 0:
        return out

    lengths = sorted(bucket_words_by_len.keys())
    for L in lengths:
        short_ws = bucket_words_by_len[L]
        if not short_ws:
            continue
        kmax_short = math.floor(max_ratio * L)
        if kmax_short <= 0:
            continue  # short side allows no edits → no cross-length pairing

        # Only k=1 permitted by INS+DEL<=1
        for k in range(1, min(1, kmax_short) + 1):
            long_ws = bucket_words_by_len.get(L + k, [])
            if not long_ws:
                continue

            R = kmax_short - k  # extra wildcards (potential SUBs) after 1 deletion
            pat2longs: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)

            for wlong in long_ws:
                q = lex[wlong]
                for base_seq in deletion_keys_k(q, k, cap_per_k):
                    for r in range(0, R + 1):
                        for pat in wildcard_patterns_exact_k(base_seq, r, cap_per_k):
                            pat2longs[pat].append((wlong, base_seq))

            seen_pairs_in_bucket = set()
            for wshort in short_ws:
                p = lex[wshort]
                for r in range(0, R + 1):
                    for pat in wildcard_patterns_exact_k(p, r, cap_per_k):
                        cand = pat2longs.get(pat, [])
                        if not cand:
                            continue
                        for wlong, _ in cand:
                            if (wshort, wlong) in seen_pairs_in_bucket:
                                continue
                            seen_pairs_in_bucket.add((wshort, wlong))

                            q = lex[wlong]

                            # short -> long (source length = len(p))
                            Dpair_s = math.floor(max_ratio * len(p))
                            if Dpair_s > 0:
                                ops, dist = align_ops_leqD(p, q, Dpair_s)
                                if (ops is not None and dist > 0 and
                                    ops_allowed(ops, allow_sub, allow_ins, allow_del)):
                                    if len(out[wshort]) < max_neighbors_per_word:
                                        out[wshort].append(make_entry(wlong, q, ops, src_len=len(p)))

                            # long -> short (source length = len(q))
                            Dpair_l = math.floor(max_ratio * len(q))
                            if Dpair_l > 0:
                                opsr, distr = align_ops_leqD(q, p, Dpair_l)
                                if (opsr is not None and distr > 0 and
                                    ops_allowed(opsr, allow_sub, allow_ins, allow_del)):
                                    if len(out[wlong]) < max_neighbors_per_word:
                                        out[wlong].append(make_entry(wshort, p, opsr, src_len=len(q)))

    for w in out:
        out[w].sort(key=lambda e: (e["distance"], e["neighbor"]))
    return out

# ---------- Orchestration ----------
def chunk_by_length(lex: Dict[str, List[str]]) -> Dict[int, List[str]]:
    buckets = defaultdict(list)
    for w, p in lex.items():
        buckets[len(p)].append(w)
    return buckets

def deduplicate_neighbors(neighbors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); unique = []
    for entry in neighbors:
        key = entry["neighbor"]
        if key not in seen:
            seen.add(key); unique.append(entry)
    return unique

def run_fast_neighbors(lex: Dict[str, List[str]],
                       allow_sub: bool,
                       allow_ins: bool,
                       allow_del: bool,
                       max_ratio: float,
                       conf_pairs: Set[Tuple[str,str]] | None,
                       max_neighbors_per_word: int,
                       cap_per_k: int | None,
                       workers: int = max(1, mp.cpu_count()//2)) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {
        w: {"canonical_pron": lex[w], "neighbors": []} for w in lex
    }
    buckets = chunk_by_length(lex)

    # Same-length neighbors (parallel per length bucket)
    args = []
    for _, ws in buckets.items():
        args.append((ws, lex, max_ratio, allow_sub, allow_ins, allow_del,
                     conf_pairs, cap_per_k, max_neighbors_per_word))
    with mp.Pool(processes=workers) as pool:
        parts = pool.starmap(same_length_neighbors, args)
    for part in parts:
        for w, entries in part.items():
            result[w]["neighbors"].extend(entries)

    # Cross-length neighbors (only if INS or DEL operations are allowed)
    if allow_ins or allow_del:
        cross_map = cross_length_neighbors(
            buckets, lex, max_ratio, allow_sub, allow_ins, allow_del,
            cap_per_k, max_neighbors_per_word
        )
        for w, entries in cross_map.items():
            result[w]["neighbors"].extend(entries)

    # Deduplicate and sort final neighbor lists
    for w in result:
        result[w]["neighbors"] = deduplicate_neighbors(result[w]["neighbors"])
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

    # ratio upper bound
    ap.add_argument("--max-ratio", type=float, default=0.2,
                    help="Upper bound on distance/length (e.g., 0.2 ⇒ <= floor(0.2*len(source))).")

    # extras
    ap.add_argument("--confusions", default="", help="Optional whitelist: 'a b' per line for pure SUB (dist=1) at same length.")
    ap.add_argument("--max-neighbors-per-word", type=int, default=50)
    ap.add_argument("--cap-per-k", type=int, default=None,
                    help="Cap patterns per order (wildcards/deletions) to avoid combinatorial explosion. Default: unlimited.")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()//2))
    return ap.parse_args()

def main():
    args = parse_args()

    try:
        lex = load_lexicon(args.lexicon)
        print(f"Loaded {len(lex)} words from lexicon", file=sys.stderr)
    except Exception as e:
        print(f"Failed to load lexicon: {e}", file=sys.stderr)
        sys.exit(1)

    # Load optional confusion pairs whitelist
    conf_pairs: Set[Tuple[str,str]] | None = None
    if args.confusions:
        conf_pairs = set()
        try:
            with open(args.confusions, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    parts = ln.split()
                    if len(parts) >= 2:
                        conf_pairs.add((parts[0], parts[1]))
            print(f"Loaded {len(conf_pairs)} confusion pairs", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to load confusions: {e}", file=sys.stderr)

    neighbors = run_fast_neighbors(
        lex=lex,
        allow_sub=args.allow_sub,
        allow_ins=args.allow_ins,
        allow_del=args.allow_del,
        max_ratio=args.max_ratio,
        conf_pairs=conf_pairs,
        max_neighbors_per_word=args.max_neighbors_per_word,
        cap_per_k=args.cap_per_k,
        workers=args.workers
    )

    # Write output
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(neighbors, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)

    # Summary statistics
    total_neighbors = sum(len(v["neighbors"]) for v in neighbors.values())
    avg_neighbors = total_neighbors / len(neighbors) if neighbors else 0

    print(f"[OK] Wrote: {args.out}", file=sys.stderr)
    print(f"  Words: {len(lex)}", file=sys.stderr)
    print(f"  Operations: sub={args.allow_sub}, ins={args.allow_ins}, del={args.allow_del}", file=sys.stderr)
    print(f"  Max ratio: {args.max_ratio}", file=sys.stderr)
    print(f"  Total neighbor pairs: {total_neighbors}", file=sys.stderr)
    print(f"  Average neighbors per word: {avg_neighbors:.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()