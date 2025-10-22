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
import argparse
import json
import multiprocessing as mp
import sys
from itertools import combinations

# Constants
SEP = " "
OP_SUB = "SUB"
OP_INS = "INS"
OP_DEL = "DEL"
OP_MATCH = "M"

# ---------------- I/O ----------------
def load_lexicon(path: str) -> Dict[str, List[str]]:
    """Load lexicon from file (JSON or text format).
    
    Args:
        path: Path to lexicon file
        
    Returns:
        Dictionary mapping words to phone sequences
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If JSON file is malformed
    """
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
    """Count occurrences of each operation type."""
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
               ops_strings: List[str]) -> Dict[str, Any]:
    """Create a neighbor entry with distance and operation details."""
    distance = sum(1 for op in ops_strings if op in (OP_SUB, OP_INS, OP_DEL))
    return {
        "neighbor": neighbor_word,
        "candidate_pron": candidate_pron,
        "distance": distance,
        "counts": count_ops_from_ops_list(ops_strings),
        "ops": ops_strings
    }

def ops_allowed(ops: List[str], allow_sub: bool, allow_ins: bool, allow_del: bool) -> bool:
    """Check if all operations in ops are allowed by the given flags."""
    for op in ops:
        if op == OP_SUB and not allow_sub:
            return False
        if op == OP_INS and not allow_ins:
            return False
        if op == OP_DEL and not allow_del:
            return False
    return True

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
    """Compute Levenshtein distance and alignment operations.
    
    Returns (ops, dist) if dist <= D, else (None, dist).
    Optimized: single DP pass with immediate traceback.
    """
    lp, lq = len(p), len(q)
    if abs(lp - lq) > D:
        return None, D + 1

    # Single full DP pass
    dp = [[0] * (lq + 1) for _ in range(lp + 1)]
    for i in range(lp + 1):
        dp[i][0] = i
    for j in range(lq + 1):
        dp[0][j] = j
    
    for i in range(1, lp + 1):
        for j in range(1, lq + 1):
            match_cost = 0 if p[i-1] == q[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j-1] + match_cost,  # substitution or match
                dp[i-1][j] + 1,              # deletion
                dp[i][j-1] + 1               # insertion
            )
    
    dist = dp[lp][lq]
    if dist > D:
        return None, dist

    # Traceback to get operations
    ops: List[str] = []
    i, j = lp, lq
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_cost = 0 if p[i-1] == q[j-1] else 1
            if dp[i][j] == dp[i-1][j-1] + match_cost:
                ops.append(OP_MATCH if p[i-1] == q[j-1] else OP_SUB)
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(OP_DEL)
            i -= 1
        else:
            ops.append(OP_INS)
            j -= 1
    
    ops.reverse()
    return ops, dist

# ---------- same-length neighbors via k-wildcards ----------
def same_length_neighbors(ws: List[str],
                          lex: Dict[str, List[str]],
                          Dmax: int,
                          allow_sub: bool, allow_ins: bool, allow_del: bool,
                          conf_pairs: Set[Tuple[str,str]] | None,
                          cap_per_k: int | None,
                          max_neighbors_per_word: int) -> Dict[str, List[Dict[str, Any]]]:
    """Find neighbors with same pronunciation length using wildcard patterns.
    
    Uses k-wildcard patterns to group candidate pairs, then validates with DP.
    """
    out = {w: [] for w in ws}
    if not ws or Dmax <= 0:
        return out

    # Build wildcard pattern index for k=1..Dmax
    pat2words: Dict[str, List[str]] = defaultdict(list)
    for w in ws:
        p = lex[w]
        for pat in wildcard_patterns_k(p, Dmax, cap_per_k):
            pat2words[pat].append(w)

    # Process candidate pairs within each bucket
    seen_pairs = set()
    for group in pat2words.values():
        if len(group) < 2:
            continue
        
        for i in range(len(group)):
            w1 = group[i]
            p1 = lex[w1]
            for j in range(i + 1, len(group)):
                w2 = group[j]
                p2 = lex[w2]
                
                if (w1, w2) in seen_pairs:
                    continue
                seen_pairs.add((w1, w2))
                
                ops, dist = align_ops_leqD(p1, p2, Dmax)
                if ops is None or not (0 < dist <= Dmax):
                    continue
                
                # Filter by allowed operation types
                if not ops_allowed(ops, allow_sub, allow_ins, allow_del):
                    continue
                
                # Optional confusion pair whitelist for pure single SUB
                if (dist == 1 and ops.count(OP_SUB) == 1 and 
                    ops.count(OP_INS) == 0 and ops.count(OP_DEL) == 0 and conf_pairs):
                    kpos = next(k for k, o in enumerate(ops) if o == OP_SUB)
                    if ((p1[kpos], p2[kpos]) not in conf_pairs and 
                        (p2[kpos], p1[kpos]) not in conf_pairs):
                        continue
                
                # Add bidirectional neighbors with capacity check
                if len(out[w1]) < max_neighbors_per_word:
                    out[w1].append(make_entry(w2, p2, ops))
                if len(out[w2]) < max_neighbors_per_word:
                    out[w2].append(make_entry(w1, p1, ops))
    
    # Sort neighbors by distance, then name
    for w in out:
        out[w].sort(key=lambda e: (e["distance"], e["neighbor"]))
    
    return out

# ---------- cross-length neighbors with delete+k-wildcards (supports INS+SUB etc.) ----------
def cross_length_neighbors(bucket_words_by_len: Dict[int, List[str]],
                           lex: Dict[str, List[str]],
                           Dmax: int,
                           allow_sub: bool, allow_ins: bool, allow_del: bool,
                           cap_per_k: int | None,
                           max_neighbors_per_word: int) -> Dict[str, List[Dict[str, Any]]]:
    """Find neighbors with different pronunciation lengths.
    
    Strategy:
      - For length difference k in 1..Dmax:
        - Delete k phones from longer words to get base sequences
        - Add r wildcards (r=0..Dmax-k) to create patterns
        - Match with shorter words using same r-wildcard patterns
        - Validate with full DP and operation type filters
    
    This approach supports mixed operations like INS+SUB.
    """
    out = {w: [] for _, ws in bucket_words_by_len.items() for w in ws}
    if Dmax <= 0:
        return out

    lengths = sorted(bucket_words_by_len.keys())
    
    for L in lengths:
        short_ws = bucket_words_by_len[L]
        if not short_ws:
            continue

        for k in range(1, Dmax + 1):
            long_ws = bucket_words_by_len.get(L + k, [])
            if not long_ws:
                continue

            # Build pattern index: pattern -> list of (wlong, base_seq_after_del)
            pat2longs: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
            
            for wlong in long_ws:
                q = lex[wlong]
                # Enumerate deletion variants (remove k phones)
                for base_seq in deletion_keys_k(q, k, cap_per_k):
                    R = Dmax - k
                    # Add r wildcards to each deletion variant
                    for r in range(0, R + 1):
                        for pat in wildcard_patterns_exact_k(base_seq, r, cap_per_k):
                            pat2longs[pat].append((wlong, base_seq))

            # Probe with shorter words using r-wildcard patterns
            # Track seen pairs to avoid duplicates
            seen_pairs_in_bucket = set()
            
            for wshort in short_ws:
                p = lex[wshort]
                R = Dmax - k
                for r in range(0, R + 1):
                    for pat in wildcard_patterns_exact_k(p, r, cap_per_k):
                        cand = pat2longs.get(pat, [])
                        if not cand:
                            continue
                        
                        for wlong, base_seq in cand:
                            # Skip if we've already processed this pair
                            if (wshort, wlong) in seen_pairs_in_bucket:
                                continue
                            seen_pairs_in_bucket.add((wshort, wlong))
                            
                            q = lex[wlong]
                            
                            # Validate short->long direction
                            ops, dist = align_ops_leqD(p, q, Dmax)
                            if (ops is not None and 0 < dist <= Dmax and 
                                ops_allowed(ops, allow_sub, allow_ins, allow_del)):
                                if len(out[wshort]) < max_neighbors_per_word:
                                    out[wshort].append(make_entry(wlong, q, ops))
                            
                            # Validate long->short direction
                            opsr, distr = align_ops_leqD(q, p, Dmax)
                            if (opsr is not None and 0 < distr <= Dmax and 
                                ops_allowed(opsr, allow_sub, allow_ins, allow_del)):
                                if len(out[wlong]) < max_neighbors_per_word:
                                    out[wlong].append(make_entry(wshort, p, opsr))

    # Sort neighbors by distance, then name
    for w in out:
        out[w].sort(key=lambda e: (e["distance"], e["neighbor"]))
    
    return out

# ---------- Orchestration ----------
def chunk_by_length(lex: Dict[str, List[str]]) -> Dict[int, List[str]]:
    """Group words by pronunciation length."""
    buckets = defaultdict(list)
    for w, p in lex.items():
        buckets[len(p)].append(w)
    return buckets

def deduplicate_neighbors(neighbors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate neighbor entries, keeping the first occurrence."""
    seen = set()
    unique = []
    for entry in neighbors:
        key = entry["neighbor"]
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique

def run_fast_neighbors(lex: Dict[str, List[str]],
                       allow_sub: bool,
                       allow_ins: bool,
                       allow_del: bool,
                       max_distance: int,
                       conf_pairs: Set[Tuple[str,str]] | None,
                       max_neighbors_per_word: int,
                       cap_per_k: int | None,
                       workers: int = max(1, mp.cpu_count()//2)) -> Dict[str, Dict[str, Any]]:
    """Main orchestration function to find all neighbors.
    
    Args:
        lex: Word to pronunciation mapping
        allow_sub/allow_ins/allow_del: Operation type filters
        max_distance: Maximum Levenshtein distance
        conf_pairs: Optional confusion pair whitelist
        max_neighbors_per_word: Capacity limit per word
        cap_per_k: Pattern generation limit per order
        workers: Number of parallel workers
        
    Returns:
        Dictionary with canonical pronunciations and neighbor lists
    """
    result: Dict[str, Dict[str, Any]] = {
        w: {"canonical_pron": lex[w], "neighbors": []} for w in lex
    }
    buckets = chunk_by_length(lex)
    Dmax = max(1, int(max_distance))

    # Same-length neighbors (parallel per length bucket)
    args = []
    for _, ws in buckets.items():
        args.append((ws, lex, Dmax, allow_sub, allow_ins, allow_del, 
                    conf_pairs, cap_per_k, max_neighbors_per_word))
    
    with mp.Pool(processes=workers) as pool:
        parts = pool.starmap(same_length_neighbors, args)
    
    for part in parts:
        for w, entries in part.items():
            result[w]["neighbors"].extend(entries)

    # Cross-length neighbors (only if INS or DEL operations are allowed)
    if allow_ins or allow_del:
        cross_map = cross_length_neighbors(
            buckets, lex, Dmax, allow_sub, allow_ins, allow_del,
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
    """Main entry point for CLI."""
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
                    # Confusions only used in same-length pure SUB=1 scenario
                    if not ln or ln.startswith("#"):
                        continue
                    parts = ln.split()
                    if len(parts) >= 2:
                        a, b = parts[0], parts[1]
                        conf_pairs.add((a, b))
            print(f"Loaded {len(conf_pairs)} confusion pairs", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to load confusions: {e}", file=sys.stderr)

    # Run neighbor finding
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
    print(f"  Max distance: {args.max_distance}", file=sys.stderr)
    print(f"  Total neighbor pairs: {total_neighbors}", file=sys.stderr)
    print(f"  Average neighbors per word: {avg_neighbors:.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()