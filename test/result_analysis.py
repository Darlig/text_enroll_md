#!/usr/bin/env python3

import sys
import os

ignore_keyword = ["noise", "NOISE"]

def load_pos_keyword(pos_keyword, ignore_keyword):
  pos_keyword_dict = {}
  #pos_ids = []
  with open(pos_keyword) as f_keyword:
      for line in f_keyword.readlines():
          keyword, id = line.strip().split()
          if keyword in ignore_keyword:
              continue
          pos_keyword_dict[keyword] = id
          #pos_ids.append(id)
  return pos_keyword_dict

def load_neg_keyword(neg_keyword, ignore_keyword):
  #neg_keyword_dict = {}
  neg_ids = []
  with open(neg_keyword) as f_keyword:
      for line in f_keyword.readlines():
          keyword, id = line.strip().split()
          if keyword in ignore_keyword:
              continue
          #neg_keyword_dict[keyword] = id
          neg_ids.append(id)
  return neg_ids

def load_pos_refer(pos_refer, pos_keyword_dict):
  pos_refer_dict = {}
  unknown_keywords = []
  with open(pos_refer) as f_pos_refer:
      for line in f_pos_refer.readlines():
          uttid, keyword = line.strip().split()
          if keyword not in pos_keyword_dict:
              if keyword not in unknown_keywords:
                  unknown_keywords.append(keyword)
              continue
          pos_refer_dict[uttid] = pos_keyword_dict[keyword]
  print("unknown_keywords:", unknown_keywords)
  return pos_refer_dict

def load_pos_scores(pos_result, pos_refer_dict):
    pos_scores_dict = {}
    with open(pos_result) as f_pos_result:
        for line in f_pos_result.readlines():
            line_sp = line.strip().split()
            uttid = line_sp[0]
            id = line_sp[2]
            score = float(line_sp[3])
            if uttid not in pos_refer_dict:
                continue
            if uttid not in pos_scores_dict:
                pos_scores_dict[uttid] = 0.0
            if pos_refer_dict[uttid] == id:
                pos_scores_dict[uttid] = max(score, pos_scores_dict[uttid])
    pos_scores = list(pos_scores_dict.values())
    pos_scores.sort()
    #print(len(pos_scores))
    #for i in range(10):
    #    print(pos_scores[i])
    return pos_scores

def load_neg_scores(neg_result, neg_ids):
    neg_scores = []
    with open(neg_result) as f_neg_result:
        for line in f_neg_result.readlines():
            line_sp = line.strip().split()
            id = line_sp[2]
            score = float(line_sp[3])
            if id in neg_ids:
                neg_scores.append(score)
            else:
                neg_scores.append(0.0)
    neg_scores.sort(reverse=True)
    #print(len(neg_scores))
    #for i in range(10):
    #    print(neg_scores[i])
    return neg_scores

