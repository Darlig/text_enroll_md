import torch
import numpy as np

class HypothesisList(object):
    def __init__(self, data):
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self):
        return self._data
    
    def add(self, hyp):
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]
            old_hyp.score = old_hyp.socre.append(hyp.score)
        else:
            self._data[key] = hyp
    
    def get_best(self):
        return max(
            self._data.valuse(), key = lambda hyp: hyp.score
        )
    
    def remove(self, hyp):
        key = hyp.key
        del self._data[key]
    
    def filter(self, th):
        new_hyp_list = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.score > th:
                new_hyp_list.add(hyp)
        return new_hyp_list

    def topN(self, n):
        hyps = list(self._data.items())
        hyps = sorted(hyps, key=lambda h: h[1].score, reverse=True)[:n]

        tK_list = HypothesisList(dict(hyps))
        return tK_list
    
    def __contains__(self, key):
        return key in iter(self._data)
    
    def __iter__(self):
        return iter(self._data.values())
    
    def __str__(self):
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)


class Hypothesis:
    ys = []
    score = 0.0
    def key(self):
        return "_".join(map(str, self.ys))

def beam_search(
    target,
    beam=5,
    blank_id=None,
    max_leng=20
):
    assert(
        isinstance(target, T)
        for T in [list, torch.Tensor, np.ndarray]
    )
    L = len(target)
    num_output = 0
    l = 0

    hyp = HypothesisList()
    hyp.add(Hypothesis(ys=[blank_id], score=0.0))
    beam_cache = {}

    while l < L and num_output < max_leng:
        c_target = target[:, l:l+1, :]
        hyp_tmp = hyp    
        B = HypothesisList()
        cache = {}
        while True:
            y_head = hyp_tmp.get_best()
            hyp_tmp.remove(y_head)
            cache_key = y_head.key

            if cache_key not in beam_cache:
                pass
